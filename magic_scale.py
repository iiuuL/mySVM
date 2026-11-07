"""MAGIC scale experiment. Run with the project venv on Windows."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from libsvm.svmutil import svm_predict
from run_tricks import official
from svm_fast import FastSVM
from svm import SVM

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/scale_magic"
SIZES = (200, 500, 1000, 2000, 5000, 10000)
BUDGET = 120

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf8")

def make_splits():
    old = np.load(ROOT / "results/baseline/magic_500_split.npz")
    y = np.load(ROOT / "data/magic.npz")["y"]
    test, current = old["test_indices"], old["train_indices"]
    remaining = np.setdiff1d(np.arange(len(y)), np.r_[test, current])
    small, _ = train_test_split(current, train_size=200, random_state=42, stratify=y[current])
    subsets = {"200":small, "500":current}
    for n in SIZES[2:]:
        extra, remaining = train_test_split(remaining, train_size=n-len(current),
                                            stratify=y[remaining], random_state=42)
        current = np.r_[current, extra]
        subsets[str(n)] = current
    for n, idx in subsets.items():
        assert len(np.unique(idx)) == int(n) and not np.intersect1d(idx,test).size
    np.savez(OUT/"indices.npz",test=test,**subsets)

def peak_mib():
    class Counters(ctypes.Structure):
        _fields_ = [("cb",wintypes.DWORD),("faults",wintypes.DWORD)] + [
            (name,ctypes.c_size_t) for name in
            ("peak","working","pool1","pool2","pool3","pool4","page","peak_page")]
    values = Counters()
    values.cb = ctypes.sizeof(values)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    read_counters = ctypes.windll.psapi.GetProcessMemoryInfo
    read_counters.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not read_counters(current(),ctypes.byref(values),values.cb):
        raise ctypes.WinError()
    return values.peak / 2**20

def worker(n, name, path):
    raw = np.load(ROOT/"data/magic.npz")
    idx = np.load(OUT/"indices.npz")
    train,test = idx[str(n)],idx["test"]
    features,labels=raw["X"],raw["y"]
    y, target = labels[train],labels[test]
    scaler=StandardScaler().fit(features[train])
    X,T=scaler.transform(features[train]),scaler.transform(features[test])
    gamma=1/X.shape[1]
    start=perf_counter()
    if name in ("v3","v0"):
        solver = FastSVM if name=="v3" else SVM
        model=solver(C=1,gamma=gamma).fit(X,y)
    else:
        model=official(X,y,1,gamma)
    seconds=perf_counter()-start
    peak=peak_mib()
    if name in ("v3","v0"):
        alpha,bias=model.alpha_,model.b_
        pred=np.concatenate([model.predict(block) for block in np.array_split(T,20)])
        status, scans=model.stop_reason_,model.n_iter_
    else:
        alpha=np.zeros(n)
        alpha[np.array(model.get_sv_indices())-1]=np.abs(np.array(model.get_sv_coef()).ravel())
        bias=-model.rho[0]*model.get_labels()[0]
        pred=np.array(svm_predict(target.tolist(),T.tolist(),model,"-q")[0])
        status,scans="returned",None
    kernel=FastSVM(C=1,gamma=gamma)
    decision=np.concatenate([kernel._kernel(X[i:i+128],X) @ (alpha*y)+bias for i in range(0,n,128)])
    margin=y*decision
    kkt=float(np.maximum(np.where(alpha<1-1e-8,1-margin,0),np.where(alpha>1e-8,margin-1,0)).max())
    save(path,dict(n=n,model=name,seconds=seconds,peak_mib=peak,correct=int(sum(pred==target)),
                   n_test=len(test),f1=f1_score(target,pred),status=status,kkt=kkt,scans=scans,
                   equality=float(abs(alpha@y)),box=float(max(0,-min(alpha),max(alpha)-1))))

if __name__=="__main__":
    OUT.mkdir(exist_ok=True)
    baseline = sys.argv[1:]==["--baseline"]
    if len(sys.argv)>1 and not baseline:
        worker(int(sys.argv[1]),sys.argv[2],Path(sys.argv[3]))
    else:
        make_splits()
        for n in SIZES:
            stopped=set()
            for repeat in range(3):
                order=("v0",) if baseline else (("libsvm","v3") if repeat%2==0 else ("v3","libsvm"))
                for name in order:
                    if name in stopped:
                        continue
                    path=OUT/f"{n}_{name}_{repeat}.json"
                    if not path.exists():
                        # Use base interpreter via the venv launcher; taskkill also stops the child.
                        proc=subprocess.Popen([sys.executable,__file__,str(n),name,str(path)])
                        try:
                            proc.wait(timeout=BUDGET)
                        except subprocess.TimeoutExpired:
                            subprocess.run(["taskkill","/F","/T","/PID",str(proc.pid)],capture_output=True)
                            proc.wait()
                            save(path,dict(n=n,model=name,status="timeout",budget=BUDGET))
                        if not path.exists():
                            raise RuntimeError(f"worker failed: {path}")
                    record=json.loads(path.read_text())
                    if record["status"] not in ("returned","converged"):
                        stopped.add(name)
                    print(n,name,repeat,record["status"],record.get("seconds"),flush=True)
            if baseline and stopped:
                break
