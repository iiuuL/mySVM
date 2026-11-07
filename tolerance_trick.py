"""小样本容差敏感性；与固定 1e-3 的正式加速比较分开。"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from svm_fast import FastSVM

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    data=np.load(root/'results/baseline/magic_500_split.npz')
    X,y,T,target=(data[k] for k in ('X','y','T','target'))
    rows=[]
    for tol in (1e-2,1e-3,1e-4):
        for repeat in range(3):
            start=time.perf_counter()
            m=FastSVM(C=1,gamma=1/X.shape[1],kernel='rbf',tol=tol).fit(X,y)
            seconds=time.perf_counter()-start
            pred=m.predict(T); stats=m.diagnostics()
            rows.append(dict(tol=tol,repeat=repeat,train_seconds=seconds,iterations=m.n_iter_,status=m.stop_reason_,
                             accuracy=float(accuracy_score(target,pred)),f1=float(f1_score(target,pred)),**stats))
            print(tol,seconds,stats['kkt_max'],rows[-1]['accuracy'],flush=True)
    (root/'results/tricks/tolerance_magic_rbf.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
