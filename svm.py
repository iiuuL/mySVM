"""教学用二分类 C-SVC：完整核矩阵 + 简单 SMO，不依赖 sklearn。"""
import warnings
import numpy as np


class SVM:
    def __init__(self, C=1.0, kernel="rbf", gamma=1.0, tol=1e-3, max_iter=2000):
        self.C, self.kernel, self.gamma = C, kernel, gamma
        self.tol, self.max_iter = tol, max_iter

    def _kernel(self, X, Z):
        if self.kernel == "linear":
            return X @ Z.T
        distance = (X * X).sum(1)[:, None] + (Z * Z).sum(1) - 2 * X @ Z.T
        return np.exp(-self.gamma * np.maximum(distance, 0))

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if (X.ndim != 2 or y.shape != (len(X),) or X.shape[1] == 0
                or not np.isfinite(X).all() or not np.isfinite(y).all()
                or set(y) != {-1, 1}):
            raise ValueError("X 必须是有限二维数组，y 必须包含 -1、+1 两类")
        if (self.kernel not in ("linear", "rbf") or self.C <= 0
                or self.gamma <= 0 or self.tol <= 0 or self.max_iter < 1
                or not np.isfinite([self.C, self.gamma, self.tol]).all()):
            raise ValueError("检查核函数和正数参数")
        self.X_, self.y_ = X.copy(), y.copy()
        n = len(y)
        K = self._kernel(X, X)
        a = self.alpha_ = np.zeros(n)
        self.b_ = 0.0
        self.n_updates_ = 0
        self.stop_reason_ = "max_iter"

        # 每轮扫描全部样本；每次只修改 alpha_i、alpha_j 和偏置 b。
        for iteration in range(self.max_iter):
            changed = 0
            for i in range(n):
                E = K @ (a * y) + self.b_ - y  # 基础版直接重算，无误差缓存
                r = y[i] * E[i]
                if not ((r < -self.tol and a[i] < self.C)
                        or (r > self.tol and a[i] > 0)):
                    continue
                # 按固定顺序尝试第二个变量，避免引入复杂工作集策略。
                for j in range(n):
                    if i == j:
                        continue
                    ai, aj = a[i], a[j]
                    s = y[i] * y[j]
                    if s == -1:
                        L, H = max(0, aj - ai), min(self.C, self.C + aj - ai)
                    else:
                        L, H = max(0, ai + aj - self.C), min(self.C, ai + aj)
                    if H <= L:
                        continue
                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    slope = y[j] * (E[i] - E[j])
                    if eta > 1e-12:
                        new_aj = np.clip(aj + slope / eta, L, H)
                    else:
                        # 重复样本可能使曲率为零；比较两个端点的对偶收益。
                        candidates = np.array([L, H])
                        delta = candidates - aj
                        new_aj = candidates[np.argmax(slope * delta - eta * delta**2 / 2)]
                    dj = new_aj - aj
                    if abs(dj) < 1e-12:
                        continue
                    di = -s * dj
                    b1 = self.b_ - E[i] - y[i] * di * K[i, i] - y[j] * dj * K[i, j]
                    b2 = self.b_ - E[j] - y[i] * di * K[i, j] - y[j] * dj * K[j, j]
                    a[i], a[j] = ai + di, new_aj
                    edge = min(1e-8, self.C * 1e-6)
                    if edge < a[i] < self.C - edge:
                        self.b_ = b1
                    elif edge < a[j] < self.C - edge:
                        self.b_ = b2
                    else:
                        self.b_ = (b1 + b2) / 2
                    changed += 1
                    break
            # 从所有样本重新确定 b，兼顾没有自由支持向量的情形。
            edge = min(1e-8, self.C * 1e-6)
            target = y - K @ (a * y)
            free = (a > edge) & (a < self.C - edge)
            if free.any():
                self.b_ = float(target[free].mean())
            else:
                lower = ((y == 1) & (a < self.C - edge)) | ((y == -1) & (a > edge))
                upper = ((y == -1) & (a < self.C - edge)) | ((y == 1) & (a > edge))
                self.b_ = float((target[lower].max() + target[upper].min()) / 2)
            self.n_updates_ += changed
            self.n_iter_ = iteration + 1
            self.diagnostics_ = self.diagnostics()
            if self.diagnostics_["kkt_max"] <= self.tol:
                self.stop_reason_ = "converged"
                break
            if changed == 0:
                self.stop_reason_ = "stalled"
                break
        self.support_ = np.flatnonzero(a > 1e-8)
        if self.stop_reason_ != "converged":
            warnings.warn(f"SVM 未收敛：{self.stop_reason_}", RuntimeWarning)
        return self

    def decision_function(self, X):
        return self._kernel(np.asarray(X, dtype=float), self.X_) @ (self.alpha_ * self.y_) + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def diagnostics(self):
        """独立重算训练残差；目标为 1/2||w||² + C * sum(hinge)。"""
        a, y = self.alpha_, self.y_
        f = self.decision_function(self.X_)
        margin = y * f
        eps = min(1e-8, self.C * 1e-6)
        violation = np.maximum(
            np.where(a < self.C - eps, np.maximum(1 - margin, 0), 0),
            np.where(a > eps, np.maximum(margin - 1, 0), 0))
        norm2 = (a * y) @ (f - self.b_)
        primal = norm2 / 2 + self.C * np.maximum(1 - margin, 0).sum()
        dual = a.sum() - norm2 / 2
        return dict(kkt_max=float(violation.max()), equality_error=float(abs(a @ y)),
                    box_error=float(max(0, -a.min(), a.max() - self.C)),
                    primal=float(primal), dual=float(dual), gap=float(primal - dual))
