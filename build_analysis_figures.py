"""仅分析已有结果，不重新训练。生成六组补充图及派生数值记录。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report"
rows = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
labels = ["乳腺癌·线性", "乳腺癌·RBF", "Banknote·线性", "Banknote·RBF",
          "Spambase·线性", "Spambase·RBF", "MAGIC 500·线性", "MAGIC 500·RBF"]
models = ["baseline", "v1", "v2", "libsvm"]
names = ["V0", "V1", "V2", "LIBSVM"]
colors = ["#8696a7", "#3795b0", "#e59a3b", "#7b61a8"]
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10})
derived = {}

def row(c, m):
    return next(r for r in rows if r["case"] == c and r["model"] == m)

def valid(r):
    return r["status"] in ("converged", "returned")

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)

# 图3：矩阵直接显示绝对值，避免将非常小的差异夸大为巨大柱高差。
fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
for ax, metric, title in zip(axes, ["accuracy", "f1"], ["Accuracy（%）", "F1（+1 类）"]):
    values = np.array([[row(c,m).get(metric,np.nan) if valid(row(c,m)) else np.nan for m in models] for c in range(8)])
    ax.imshow(values, cmap="YlGnBu", vmin=.75, vmax=1, aspect="auto")
    ax.set_xticks(range(4), names)
    ax.set_yticks(range(8),labels)
    ax.set_title(title)
    for c in range(8):
        for j,m in enumerate(models):
            value = values[c,j]
            txt = (f"{value*100:.2f}" if metric=="accuracy" else f"{value:.4f}") if np.isfinite(value) else ("超时" if row(c,m)["status"]=="timeout" else "未收敛")
            ax.text(j,c,txt,ha="center",va="center",color="white" if np.isfinite(value) and value>.92 else "#333333")
fig.suptitle("分类效果：仅将达到收敛要求的结果纳入数值比较",fontsize=14)
save(fig,"classification_comparison")

# 图4：对数加速比点图，不给超时和未收敛配置填充虚假加速比。
fig,ax=plt.subplots(figsize=(11,5.5),layout="constrained")
pairs=[("baseline","v1","V0 / V1"),("v1","v2","V1 / V2"),("baseline","v2","V0 / V2")]
speedups=[]
for j,(a,b,label) in enumerate(pairs):
    for c in range(8):
        if valid(row(c,a)) and valid(row(c,b)):
            v=row(c,a)["time"]/row(c,b)["time"]
            ax.scatter(v,c+(j-1)*.19,s=55,color=colors[j],label=label if c==0 else None)
            ax.annotate(f"{v:.1f}×",(v,c+(j-1)*.19),xytext=(5,0),textcoords="offset points",va="center",fontsize=8)
            speedups.append(dict(case=c,comparison=label,ratio=v))
ax.axvline(1,color="#555",ls="--",lw=1)
ax.set_xscale("log");ax.set_xlim(.8,130)
ax.set_yticks(range(8),labels);ax.invert_yaxis()
ax.set_xlabel("平均训练耗时之比（右侧版本相对左侧版本的加速倍数）")
ax.set_title("哪些改动带来了加速？空白表示无法作充分收敛的成对比较")
ax.grid(axis="x",alpha=.2);ax.legend(loc="center right",bbox_to_anchor=(.99,.39))
derived["speedups"]=speedups
save(fig,"speedup_comparison")

# 图5：只对四个三个版本均收敛的案例计算总时间/扫描数。
cases=[0,1,2,7]
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
derived["time_per_scan"]=[]
for j,m in enumerate(models[:3]):
    x=np.arange(4)+(j-1)*.24
    ns=[row(c,m)["iterations"] for c in cases]
    ms=[1000*row(c,m)["time"]/row(c,m)["iterations"] for c in cases]
    axes[0].bar(x,ns,.22,label=names[j],color=colors[j])
    axes[1].bar(x,ms,.22,label=names[j],color=colors[j])
    derived["time_per_scan"] += [dict(case=c,model=m,milliseconds=v) for c,v in zip(cases,ms)]
for ax,title in zip(axes,["扫描轮数","总训练时间 / 扫描轮数（毫秒）"]):
    ax.set_xticks(range(4),[labels[c].replace("·","\n") for c in cases]);ax.set_yscale("log")
    ax.set_title(title);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True);ax.legend()
fig.suptitle("时间下降来自哪里？右图是含初始化与诊断的摊销量，不是独立测得的循环耗时",fontsize=12)
save(fig,"cost_decomposition")

# 图6：状态矩阵 + MAGIC 线性核最终残差；不伪造收敛轨迹。
fig,axes=plt.subplots(1,2,figsize=(12,5.7),layout="constrained",gridspec_kw={"width_ratios":[1.4,1]})
codes={"converged":0,"returned":0,"max_iter":1,"timeout":2}
states=np.array([[codes[row(c,m)["status"]] for m in models] for c in range(8)])
axes[0].imshow(states,cmap=ListedColormap(["#d4ecdf","#f8dfad","#efc7c7"]),vmin=0,vmax=2,aspect="auto")
axes[0].set_xticks(range(4),names);axes[0].set_yticks(range(8),labels);axes[0].set_title("配置完成情况（每格一个数据集×核×版本）")
for c in range(8):
    for j in range(4):axes[0].text(j,c,["达标","迭代上限","超时"][states[c,j]],ha="center",va="center")
values=[row(6,m)["kkt_max"] for m in models]
axes[1].bar(names,values,color=colors)
axes[1].axhline(.001,color="#a33",ls="--",label="要求：0.001")
axes[1].set_yscale("log");axes[1].set_ylim(.0003,.12)
axes[1].set_title("MAGIC 线性核：最终最大 KKT 残差")
for j,v in enumerate(values):axes[1].text(j,v*1.12,f"{v:.5f}",ha="center",fontsize=9)
axes[1].legend();axes[1].grid(axis="y",alpha=.2);axes[1].set_axisbelow(True)
save(fig,"convergence_status")

# 图7：已有边界样本决策值。
case=json.loads((ROOT/"results/optimization/boundary_case.json").read_text(encoding="utf-8"))[0]
fig,ax=plt.subplots(figsize=(9,3.8),layout="constrained")
values=[case[m] for m in models[:3]]
ax.barh(names[:3],values,color=colors[:3]);ax.axvline(0,color="#333",lw=1)
ax.set_xlim(-.00032,.00045);ax.invert_yaxis()
for j,v in enumerate(values):ax.annotate(f"{v:+.6f}",(v,j),xytext=(5 if v>0 else -5,0),textcoords="offset points",ha="left" if v>0 else "right",va="center")
ax.set_xlabel("决策函数 f(x)：正数预测 +1，负数预测 −1")
ax.set_title("乳腺癌 RBF：多预测正确的一个样本，恰好位于零附近（真实标签 +1）")
save(fig,"boundary_observation")

# 图8：支持向量数量及接口预测耗时，均仅展示收敛配置。
fig,axes=plt.subplots(1,2,figsize=(13,5.2),layout="constrained")
for j,m in enumerate(models):
    for c in range(8):
        r=row(c,m)
        if not valid(r):continue
        for ax,key,factor in [(axes[0],"support_vectors",1),(axes[1],"prediction_seconds",1000)]:
            ax.scatter(c+(j-1.5)*.15,r[key]*factor,color=colors[j],label=names[j] if c==0 else None,s=28)
for ax,title in zip(axes,["支持向量数","完整测试集接口预测耗时（毫秒）"]):
    ax.set_xticks(range(8),[x.replace("·","\n") for x in labels],fontsize=8)
    ax.set_yscale("log");ax.set_title(title);ax.grid(axis="y",alpha=.2);ax.legend(ncol=2)
fig.suptitle("训练更快是否意味着预测更快？跨数据集的测试样本数不同，仅在同一列内比较",fontsize=12)
save(fig,"prediction_and_support")
(OUT/"derived_analysis.json").write_text(json.dumps(derived,indent=2),encoding="utf-8")
print("Six supplementary PNG/SVG figures and derived_analysis.json saved")
