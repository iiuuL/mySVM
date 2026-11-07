"""V3 同期对照 + 仅训练池内三折调参。所有选择在测试评估前冻结。"""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from libsvm.svmutil import svm_train, svm_predict, svm_save_model
from svm import SVM
from svm_optimized import SVMV2
from svm_fast import FastSVM

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/tricks"
CASES = [("breast_cancer",0),("banknote",0),("spambase",0),("magic",500)]

def write(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")

def official(X,y,C,gamma,kernel="rbf"):
    return svm_train(y.tolist(),X.tolist(),f"-s 0 -t {0 if kernel=='linear' else 2} -c {C} -g {gamma} -e .001 -m 100 -h 1 -b 0 -q")

def tune(args):
    # 此函数只读取原训练池的索引与数据，不读取测试标签或测试数组。
    indices = np.load(ROOT/f"results/baseline/{args.dataset}_{args.limit}_split.npz")["train_indices"]
    raw = np.load(ROOT/f"data/{args.dataset}.npz")
    X, y = raw["X"][indices], raw["y"][indices]
    d = X.shape[1]
    folds = []
    splitter=StratifiedKFold(n_splits=3,shuffle=True,random_state=2026)
    fold_indices=[]
    for tr,va in splitter.split(X,y):
        scaler=StandardScaler().fit(X[tr])
        folds.append((scaler.transform(X[tr]),y[tr],scaler.transform(X[va]),y[va]))
        fold_indices.append(dict(train_original=indices[tr].tolist(),validation_original=indices[va].tolist()))
    records=[]
    start=time.perf_counter()
    for C,factor in itertools.product([.1,1.,10.,100.],[.01,.1,1.,10.,100.]):
        scores=[]; durations=[]
        for A,labels,B,target in folds:
            t=time.perf_counter(); m=official(A,labels,C,factor/d)
            durations.append(time.perf_counter()-t)
            pred,_,_=svm_predict(target.tolist(),B.tolist(),m,"-q")
            scores.append(float(accuracy_score(target,pred)))
        records.append(dict(C=C,gamma=factor/d,gamma_factor=factor,fold_accuracy=scores,
                            cv_accuracy=float(np.mean(scores)),cv_fit_seconds=float(np.mean(durations))))
        write(OUT/f"{args.dataset}_search_progress.json",records)
        print(args.dataset,C,factor,round(np.mean(scores),5),flush=True)
    # 确定性并列规则：精度完全相同时优先更小 C，再更小 gamma。
    best=sorted(records,key=lambda r:(-r["cv_accuracy"],r["C"],r["gamma"]))[0]
    default=next(r for r in records if r["C"]==1 and r["gamma_factor"]==1)
    # 接近最优精度的轻量参数，给精度与速度提供另一种验证集选择。
    near=[r for r in records if r["cv_accuracy"]>=best["cv_accuracy"]-.005]
    economical=sorted(near,key=lambda r:(r["C"],r["gamma"],-r["cv_accuracy"]))[0]
    write(args.output,dict(best=best,economical=economical,default=default,records=records,
                           search_seconds=time.perf_counter()-start,folds=fold_indices,
                           selection="max CV accuracy, ties smaller C then gamma; economical within 0.005, smallest C then gamma"))

def worker(args):
    split=np.load(ROOT/f"results/baseline/{args.dataset}_{args.limit}_split.npz")
    X,y,T,target=(split[k] for k in ("X","y","T","target"))
    C,gamma=1.,1/X.shape[1]
    if args.setting!="default":
        chosen=json.loads((OUT/f"{args.dataset}_selection.json").read_text(encoding="utf-8"))[args.setting]
        C,gamma=chosen["C"],chosen["gamma"]
    start=time.perf_counter()
    if args.model=="libsvm":m=official(X,y,C,gamma,args.kernel)
    else:
        cls=FastSVM if args.model=="v3" else SVMV2
        m=cls(C=C,gamma=gamma,kernel=args.kernel).fit(X,y)
    fit=time.perf_counter()-start
    def predict():
        if args.model=="libsvm":return np.asarray(svm_predict(target.tolist(),T.tolist(),m,"-q")[0],dtype=int)
        return m.predict(T)
    pred=predict()  # 预热一次，之后七次预测计时取中位数
    timings=[]
    for _ in range(7):
        start=time.perf_counter(); again=predict();timings.append(time.perf_counter()-start)
        assert np.array_equal(pred,again)
    if args.model=="libsvm":
        checker=SVM(C=C,kernel=args.kernel,gamma=gamma)
        checker.X_,checker.y_=X,y
        checker.alpha_=np.zeros(len(y))
        checker.alpha_[np.asarray(m.get_sv_indices())-1]=np.abs(np.asarray(m.get_sv_coef()).ravel())
        checker.b_=-m.rho[0]*m.get_labels()[0]
        stats=checker.diagnostics();status="returned";iterations=None
        sv=m.get_nr_sv();err=None
        svm_save_model(args.output+".model",m)
    else:
        stats=m.diagnostics();status=m.stop_reason_;iterations=m.n_iter_;sv=len(m.support_)
        reference=m._kernel(T,m.X_) @ (m.alpha_*m.y_)+m.b_
        err=float(np.max(np.abs(m.decision_function(T)-reference)))
        assert err<1e-8
        assert np.array_equal(pred,np.where(reference>=0,1,-1))
        np.savez(args.output+".model.npz",X=m.X_,y=m.y_,alpha=m.alpha_,b=m.b_,C=C,gamma=gamma,kernel=args.kernel)
    write(args.output,dict(dataset=args.dataset,limit=args.limit,kernel=args.kernel,model=args.model,setting=args.setting,
        C=C,gamma=gamma,n_train=len(y),n_test=len(target),train_seconds=fit,predict_seconds=float(np.median(timings)),
        predict_repeats=timings,accuracy=float(accuracy_score(target,pred)),f1=float(f1_score(target,pred)),
        precision=float(precision_score(target,pred)),recall=float(recall_score(target,pred)),
        status=status,iterations=iterations,sv=sv,prediction_error=err,prediction=pred.tolist(),**stats))

def launch(args, extra, output, timeout):
    if output.exists():return json.loads(output.read_text(encoding="utf-8"))
    cmd=[sys.executable,str(Path(__file__)),*extra,"--output",str(output)]
    print(output.stem,"running",flush=True)
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        stdout,stderr=process.communicate(timeout=timeout)
        if process.returncode:
            write(output,dict(status="error",stderr=stderr,stdout=stdout));print(stderr,flush=True)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True)
        process.communicate();write(output,dict(status="timeout",process_budget_seconds=timeout))
    r=json.loads(output.read_text(encoding="utf-8"))
    print(r.get("status","selected"),r.get("train_seconds"),r.get("accuracy"),flush=True)
    return r

def main(args):
    OUT.mkdir(exist_ok=True)
    protocol=dict(C_grid=[.1,1,10,100],gamma_factors=[.01,.1,1,10,100],cv=3,cv_seed=2026,
        scoring="accuracy",economical_tolerance=.005,test_selection=False,repeats=3,fit_process_budget=120,
        prediction="one warm-up then median of seven full test predictions; across-fit report mean",
        source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ["svm.py","svm_optimized.py","svm_fast.py", "run_tricks.py"]})
    if not (OUT/"protocol.json").exists():write(OUT/"protocol.json",protocol)
    # 参数选择先全部冻结，再执行测试评估，避免测试结果反过来改变搜索范围。
    for name,limit in CASES:
        result=launch(args,["--tune","--dataset",name,"--limit",str(limit)],OUT/f"{name}_selection.json",600)
        if "best" not in result:raise RuntimeError(f"Search failed: {name}")
    for name,limit in CASES:
        # 同期固定参数实验，两版本交替顺序；LIBSVM 一起重复测量。
        for kernel in ("linear","rbf"):
            failed=set()
            for repeat in range(3):
                order=["v2","v3","libsvm"] if repeat%2==0 else ["libsvm","v3","v2"]
                for model in order:
                    if model in failed:continue
                    result=launch(args,["--worker","--dataset",name,"--limit",str(limit),"--kernel",kernel,
                        "--model",model,"--setting","default"],OUT/f"{name}_{kernel}_default_{model}_{repeat}.json",120)
                    if result["status"] not in ("converged","returned"):failed.add(model)
        for setting in ("best","economical"):
            failed=set()
            for repeat in range(3):
                for model in (["v3","libsvm"] if repeat%2==0 else ["libsvm","v3"]):
                    if model in failed:continue
                    result=launch(args,["--worker","--dataset",name,"--limit",str(limit),"--kernel","rbf",
                        "--model",model,"--setting",setting],OUT/f"{name}_rbf_{setting}_{model}_{repeat}.json",120)
                    if result["status"] not in ("converged","returned"):failed.add(model)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",action="store_true");parser.add_argument("--tune",action="store_true")
    parser.add_argument("--dataset");parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--model");parser.add_argument("--kernel",default="rbf")
    parser.add_argument("--setting",default="default");parser.add_argument("--output")
    args=parser.parse_args()
    tune(args) if args.tune else worker(args) if args.worker else main(args)
