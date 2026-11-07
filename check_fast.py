"""V3 的边界回归与预测等价检查。"""
from check_svm import main
from svm_fast import FastSVM
import numpy as np

if __name__ == "__main__":
    main(FastSVM, "v3_checks.json")
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=70, noise=.1, random_state=7)
    for kernel in ("linear", "rbf"):
        m = FastSVM(kernel=kernel).fit(X, 2*y-1)
        exact = m._kernel(X, m.X_) @ (m.alpha_*m.y_) + m.b_
        np.testing.assert_allclose(m.decision_function(X), exact, atol=1e-10, rtol=1e-10)
        m.fit(X, 2*y-1)  # 重新 fit 不应使用旧压缩模型
        np.testing.assert_allclose(m.decision_function(X), exact, atol=1e-10, rtol=1e-10)
    print("V3 regression and prediction equivalence passed")
