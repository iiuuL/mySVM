"""小规模正确性检查，不是正式性能基准。"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from libsvm.svmutil import svm_train, svm_predict
from svm import SVM


def main(model_class=SVM, output_name="initial_checks.json", tol=1e-3):
    cases = []
    for name, data in [("blobs", make_blobs(n_samples=60, centers=2, random_state=7)),
                       ("moons", make_moons(n_samples=80, noise=.15, random_state=7))]:
        X, y = data
        cases.append((name, X, 2 * y - 1))
    cases.append(("duplicate_conflict", np.array([[0.], [0.], [-1.], [1.]]),
                  np.array([-1, 1, -1, 1])))
    data = np.load(Path(__file__).parent / "data/breast_cancer.npz")
    cases.append(("breast_cancer", data["X"], data["y"]))
    results = []
    for name, X, y in cases:
        if name == "duplicate_conflict":
            train, test = np.arange(len(y)), np.arange(len(y))
        else:
            train, test = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=42)
        scaler = StandardScaler().fit(X[train])
        A, B = scaler.transform(X[train]), scaler.transform(X[test])
        for kernel in ("linear", "rbf"):
            for C in (.01, 1., 10.):
                gamma = 1 / X.shape[1]
                model = model_class(C=C, kernel=kernel, gamma=gamma, tol=tol).fit(A, y[train])
                reference = svm_train(y[train].tolist(), A.tolist(),
                    f"-s 0 -t {0 if kernel == 'linear' else 2} -c {C} -g {gamma} -e 0.00001 -q")
                pred, _, values = svm_predict(y[test].tolist(), B.tolist(), reference, "-q")
                stats = model.diagnostics()
                agreement = float(np.mean(model.predict(B) == pred))
                assert model.stop_reason_ == "converged", (name, kernel, C, stats)
                assert stats["box_error"] < 1e-8 and stats["equality_error"] < 1e-7
                # 可行解的间隙受 C*n*最大 KKT 残差控制；不混用任意相对阈值。
                assert -1e-7 <= stats["gap"] <= C * len(A) * tol * 1.05 + 1e-6
                # 边界 f(x)=0 的分类约定可能不同；此处比较对齐标签后的决策值。
                if name == "duplicate_conflict":
                    reference_f = np.asarray(values).ravel() * reference.get_labels()[0]
                    assert np.max(np.abs(model.decision_function(B) - reference_f)) < .01
                else:
                    assert agreement >= .95, (name, kernel, C, agreement)
                row = dict(dataset=name, kernel=kernel, C=C, agreement=agreement,
                           accuracy=float(np.mean(model.predict(B) == y[test])),
                           iterations=model.n_iter_, stop=model.stop_reason_, **stats)
                results.append(row)
                print(name, kernel, C, "PASS", stats["kkt_max"], flush=True)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / output_name).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
