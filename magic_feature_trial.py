"""MAGIC: training-fold comparison of signed log features and V3 fit time."""
import json
import os
from pathlib import Path
from time import perf_counter

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from svm_fast import FastSVM

root = Path(__file__).resolve().parent
split = np.load(root / "results/baseline/magic_500_split.npz")
raw = np.load(root / "data/magic.npz")
train = split["train_indices"]
X, y = raw["X"][train], raw["y"][train]

def fit_score(A, labels, B, targets, config):
    transform, C, gamma = config
    if transform == "signed_log":
        A = np.sign(A) * np.log1p(np.abs(A))
        B = np.sign(B) * np.log1p(np.abs(B))
    scaler = StandardScaler().fit(A)
    A, B = scaler.transform(A), scaler.transform(B)
    start = perf_counter()
    model = FastSVM(C=C, gamma=gamma).fit(A, labels)
    duration = perf_counter() - start
    residual = model.diagnostics()["kkt_max"]
    assert residual <= .001 and model.stop_reason_ == "converged"
    return dict(seconds=duration, correct=int(sum(model.predict(B)==targets)), n=len(B), kkt=residual)

if __name__ == "__main__":
    configs = [("raw",100,.01), ("raw",1,.1)]
    configs += [("signed_log",c,g) for c in (1,10) for g in (.03,.1)]
    records = []
    for config in configs:
        folds = [fit_score(X[a],y[a],X[b],y[b],config)
                 for a,b in StratifiedKFold(3,shuffle=True,random_state=2026).split(X,y)]
        records.append(dict(config=config, folds=folds, accuracy=np.mean([r["correct"]/r["n"] for r in folds]),
                            seconds=np.mean([r["seconds"] for r in folds])))
    # Choose by training folds; require a faster fold mean than the tuned reference.
    winner = max((r for r in records[1:] if r["seconds"] < records[0]["seconds"]),key=lambda r:r["accuracy"])
    test = split["test_indices"]
    final = {}
    for repeat in range(3):
        order = [records[0],winner] if repeat%2==0 else [winner,records[0]]
        for r in order:
            config=r["config"]
            final.setdefault(str(config),[]).append(fit_score(X,y,raw["X"][test],raw["y"][test],config))
    result=dict(validation=records, selected=winner["config"], test=final)
    (root/"results/tricks/magic_feature_trial.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(result["selected"],final)
