"""固定协议的基础版/LIBSVM 对照；每次训练在独立进程运行。"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from libsvm.svmutil import svm_train, svm_predict, svm_save_model
from svm import SVM
from svm_optimized import SVMV1, SVMV2

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/baseline"


def prepare(name, limit):
    data = np.load(ROOT / f"data/{name}.npz")
    X, y = data["X"], data["y"]
    train, test = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=42)
    if limit and len(train) > limit:
        train, _ = train_test_split(train, train_size=limit, stratify=y[train], random_state=42)
    scaler = StandardScaler().fit(X[train])
    path = OUT / f"{name}_{limit}_split.npz"
    np.savez(path, X=scaler.transform(X[train]), y=y[train], T=scaler.transform(X[test]),
             target=y[test], train_indices=train, test_indices=test,
             mean=scaler.mean_, scale=scaler.scale_)
    return path


def worker(args):
    data = np.load(args.split)
    X, y, T, target = (data[k] for k in ("X", "y", "T", "target"))
    gamma = 1 / X.shape[1]
    start = time.perf_counter()
    if args.model != "libsvm":
        cls = {"baseline": SVM, "v1": SVMV1, "v2": SVMV2}[args.model]
        model = cls(kernel=args.kernel, gamma=gamma).fit(X, y)
    else:
        model = svm_train(y.tolist(), X.tolist(),
            f"-s 0 -t {0 if args.kernel == 'linear' else 2} -c 1 -g {gamma} -e .001 -m 100 -h 1 -b 0 -q")
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    if args.model != "libsvm":
        prediction = model.predict(T)
    else:
        prediction, _, _ = svm_predict(target.tolist(), T.tolist(), model, "-q")
    predict_seconds = time.perf_counter() - start
    # 使用相同公式、相同标签方向重算两种实现的训练残差。
    if args.model == "libsvm":
        checker = SVM(kernel=args.kernel, gamma=gamma)
        checker.X_, checker.y_ = X, y
        checker.alpha_ = np.zeros(len(y))
        checker.alpha_[np.asarray(model.get_sv_indices()) - 1] = np.abs(np.asarray(model.get_sv_coef()).ravel())
        checker.b_ = -model.rho[0] * model.get_labels()[0]
        diagnostic = checker.diagnostics()
        status = "returned"  # 官方接口不提供同等的逐轮停止状态。
        supports = model.get_nr_sv()
        svm_save_model(args.output + ".model", model)
        iterations = None
    else:
        diagnostic = model.diagnostics()
        status, supports, iterations = model.stop_reason_, len(model.support_), model.n_iter_
        np.savez(args.output + ".model.npz", X=X, y=y, alpha=model.alpha_, b=model.b_,
                 kernel=args.kernel, gamma=gamma, C=1.)
    row = dict(model=args.model, kernel=args.kernel, n_train=len(y), n_test=len(target),
               train_seconds=train_seconds, predict_seconds=predict_seconds,
               accuracy=accuracy_score(target, prediction), precision=precision_score(target, prediction, zero_division=0),
               recall=recall_score(target, prediction, zero_division=0), f1=f1_score(target, prediction, zero_division=0),
               support_vectors=supports, iterations=iterations, status=status, **diagnostic)
    row["prediction"] = np.asarray(prediction).astype(int).tolist()
    if args.model in ("v1", "v2"):
        row["cache_drift"] = model.cache_drift_
        row["updates"] = model.n_updates_
    Path(args.output).write_text(json.dumps(row, indent=2), encoding="utf-8")


def main(args):
    global OUT
    if args.optimize:
        OUT = ROOT / "results/optimization"
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [("breast_cancer", 0), ("banknote", 0), ("spambase", 0), ("magic", 500)]
    if args.pilot:
        cases = [("spambase", 200)]
    protocol = dict(C=1, gamma="1 / n_features", tol=.001, seed=42, test_fraction=.25,
                    repeats=args.repeats, process_timeout_seconds=args.timeout, threads=1,
                    svm_sha256=hashlib.sha256((ROOT / "svm.py").read_bytes()).hexdigest(),
                    cases=cases, python=sys.version)
    if args.optimize:
        protocol["optimized_sha256"] = hashlib.sha256((ROOT / "svm_optimized.py").read_bytes()).hexdigest()
    (OUT / ("pilot_protocol.json" if args.pilot else "protocol.json")).write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for name, limit in cases:
        split = prepare(name, limit)
        for kernel in ("linear", "rbf"):
            failed = set()
            for repeat in range(args.repeats):
                models = ("v1", "v2") if args.optimize else ("baseline", "libsvm")
                for model in (models if repeat % 2 == 0 else models[::-1]):
                    if model in failed:
                        continue
                    output = OUT / f"{name}_{limit}_{kernel}_{model}_{repeat}.json"
                    if args.resume and output.exists():
                        existing = json.loads(output.read_text(encoding="utf-8"))
                        if existing["status"] in ("timeout", "error", "stalled", "max_iter"):
                            failed.add(model)
                        continue
                    command = [sys.executable, str(Path(__file__)), "--worker", "--split", str(split),
                               "--kernel", kernel, "--model", model, "--output", str(output)]
                    print(output.stem, "running", flush=True)
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    try:
                        stdout, stderr = process.communicate(timeout=args.timeout)
                        if process.returncode:
                            output.write_text(json.dumps(dict(status="error", stderr=stderr)), encoding="utf-8")
                            failed.add(model)
                        else:
                            row = json.loads(output.read_text(encoding="utf-8"))
                            print(row["status"], row["train_seconds"], row["accuracy"], flush=True)
                            if row["status"] == "stalled" or row["status"] == "max_iter":
                                failed.add(model)
                    except subprocess.TimeoutExpired:
                        # Windows venv 启动器可能还有子进程，超时必须结束整棵进程树。
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                        process.communicate()
                        output.write_text(json.dumps(dict(status="timeout", process_timeout_seconds=args.timeout)), encoding="utf-8")
                        failed.add(model)
                        print("timeout", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--split")
    parser.add_argument("--kernel")
    parser.add_argument("--model")
    parser.add_argument("--output")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    worker(args) if args.worker else main(args)
