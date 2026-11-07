"""Build the scale figure and measured results used in report section 2.3."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
out=ROOT/"results/scale_magic"
report=ROOT/"report"
sizes=[200,500,1000,2000,5000,10000]
groups=defaultdict(list)
for path in sorted(out.glob("*_*_*.json")):
    r=json.loads(path.read_text())
    groups[r["n"],r["model"]].append(r)
def complete(n,model):
    group=groups[n,model]
    if not group or any(r["status"] not in ("converged","returned") for r in group):
        return []
    assert len(group)==3 and len({r["correct"] for r in group})==1
    assert all(r["kkt"]<=.00101 and r["equality"]<1e-6 and r["box"]<1e-7 and r["n_test"]==4755 for r in group)
    return group
def metric(n,model,key):
    return mean(r[key] for r in complete(n,model))
def cell(n,m,key):
    group=complete(n,m)
    if not group:
        return ("超时" if groups[n,m] and groups[n,m][0]["status"]=="timeout" else "—")
    vals=[r[key] for r in group]
    if key=="seconds":return f"{mean(vals):.4f} ± {stdev(vals):.4f}"
    if key=="correct":return f"{vals[0]/4755:.2%}"
    return f"{mean(vals):.0f}"
fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout="constrained")
for name,label,style in [("v0","V0","^-"),("v3","V3","o-"),("libsvm","LIBSVM","x--")]:
    measured=[n for n in sizes if complete(n,name)]
    for ax,key,mult in zip(axes,["seconds","correct","peak_mib"],[1,100/4755,1]):
        if measured:ax.plot(measured,[metric(n,name,key)*mult for n in measured],style,label=label)
for ax,title in zip(axes,["Fit time (s, log scale)","Test accuracy (%)","Peak working set (MiB)"]):
    ax.set_xscale("log")
    ax.set_xticks(sizes,["200","500","1k","2k","5k","10k"])
    ax.set_xlabel("Training samples")
    ax.set_ylabel(title)
    ax.grid(alpha=.23)
    ax.legend()
axes[0].set_yscale("log")
fig.savefig(report/"magic_scale.png",dpi=170)
plt.close(fig)
rows=["|训练样本|V0 训练秒|V3 训练秒|LIBSVM 训练秒|测试准确率（V3 / LIBSVM）|峰值 MiB（V3 / LIBSVM）|",
      "|---:|---:|---:|---:|---:|---:|"]
for n in sizes:
    rows.append("|"+"|".join([f"{n:,}",*[cell(n,m,"seconds") for m in ("v0","v3","libsvm")],
        " / ".join(cell(n,m,"correct") for m in ("v3","libsvm")),
        " / ".join(cell(n,m,"peak_mib") for m in ("v3","libsvm"))])+"|")
(out/"table.md").write_text("\n".join(rows)+"\n",encoding="utf8")
