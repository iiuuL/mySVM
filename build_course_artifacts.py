"""从实测记录生成课程报告所需汇总表及两张图。绘图使用 matplotlib。"""
from pathlib import Path
import csv
import json
import statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)
cases = [(n, limit, k) for n, limit in [("breast_cancer", 0), ("banknote", 0),
          ("spambase", 0), ("magic", 500)] for k in ("linear", "rbf")]
labels = ["乳腺癌\n线性", "乳腺癌\nRBF", "Banknote\n线性", "Banknote\nRBF",
          "Spambase\n线性", "Spambase\nRBF", "MAGIC 500\n线性", "MAGIC 500\nRBF"]
models = ["baseline", "v1", "v2", "libsvm"]
names = ["V0 基础版", "V1 误差缓存", "V2 改进变量选择", "LIBSVM"]
rows = []
for case_index, (dataset, limit, kernel) in enumerate(cases):
    for model in models:
        folder = "baseline" if model in ("baseline", "libsvm") else "optimization"
        files = sorted((ROOT / "results" / folder).glob(f"{dataset}_{limit}_{kernel}_{model}_*.json"))
        records = [json.loads(p.read_text(encoding="utf-8")) for p in files]
        good = [r for r in records if "accuracy" in r]
        row = dict(case=case_index, dataset=dataset, kernel=kernel, model=model,
                   status=records[0]["status"] if records else "missing", repeats=len(good))
        if good:
            first = good[0]
            for key in ("accuracy", "f1", "precision", "recall", "iterations", "kkt_max", "gap", "support_vectors", "n_train", "n_test"):
                row[key] = first[key]
            row["time"] = st.mean(r["train_seconds"] for r in good)
            row["sd"] = st.stdev(r["train_seconds"] for r in good) if len(good) > 1 else None
            row["prediction_seconds"] = st.mean(r["predict_seconds"] for r in good)
            row["cache_drift"] = max(r.get("cache_drift", 0) for r in good)
            assert all(r["prediction"] == first["prediction"] for r in good)
            assert all(r["iterations"] == first["iterations"] for r in good)
            if first["status"] in ("converged", "returned"):
                assert first["kkt_max"] <= .00101
                assert first["equality_error"] < 1e-6 and first["box_error"] < 1e-7
        rows.append(row)
(OUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
fields = list(dict.fromkeys(k for r in rows for k in r))
with (OUT / "performance.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

table = ["|数据／核|版本|训练秒（均值±标准差）|扫描轮数|Accuracy|F1|状态|",
         "|---|---|---:|---:|---:|---:|---|"]
for r in rows:
    title = labels[r["case"]].replace("\n", "／")
    name = names[models.index(r["model"])]
    if "time" not in r:
        table.append(f"|{title}|{name}|—|—|—|—|{r['status']}|")
    else:
        timing = f"{r['time']:.4f}" + (f" ± {r['sd']:.4f}" if r["sd"] is not None else "（单次）")
        state = {"converged":"收敛", "returned":"通过残差检查", "max_iter":"达到上限，未收敛"}.get(r["status"],r["status"])
        table.append(f"|{title}|{name}|{timing}|{r['iterations'] if r['iterations'] is not None else '—'}|{r['accuracy']:.2%}|{r['f1']:.4f}|{state}|")
(OUT / "performance_table.md").write_text("\n".join(table)+"\n",encoding="utf-8")

plt.rcParams.update({"font.family":"Microsoft YaHei", "axes.unicode_minus":False, "font.size":10})
colors = ["#8696a7", "#3795b0", "#e59a3b", "#7b61a8"]
for metric, count, ylabel, filename in [("time",4,"训练时间 / 秒（对数坐标）","training_time"),
                                        ("iterations",3,"完整扫描轮数（对数坐标）","scan_counts")]:
    fig, ax = plt.subplots(figsize=(13, 5.6))
    width = .19 if count == 4 else .24
    for j, model in enumerate(models[:count]):
        for c in range(len(cases)):
            row = next(r for r in rows if r["model"] == model and r["case"] == c)
            x = c + (j - (count - 1) / 2) * width
            if metric not in row or row[metric] is None:
                ax.text(x, .96, "超时" if row["status"] == "timeout" else "缺失",
                        transform=ax.get_xaxis_transform(), rotation=90, ha="center", va="top", fontsize=8, color=colors[j])
                continue
            ax.bar(x, row[metric], width*.9, color=colors[j],
                   hatch="///" if row["status"] == "max_iter" else None,
                   label=names[j] if c == 0 else None)
            if metric == "time" and row["sd"] is not None:
                ax.errorbar(x,row[metric], yerr=row["sd"],fmt="none",ecolor="#333333",capsize=2,lw=.7)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cases)),labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y",alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=count,loc="upper center",bbox_to_anchor=(.5,1.15),frameon=False)
    fig.text(.5,.015,"斜线柱：达到扫描上限但未收敛；超时无柱，不将 120 秒进程预算当作训练耗时。",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95))
    fig.savefig(OUT / (filename+".png"),dpi=180)
    fig.savefig(OUT / (filename+".svg"))
    plt.close(fig)
print("Saved summary, table and two figures")
