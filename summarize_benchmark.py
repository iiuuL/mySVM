"""由原始 JSON 生成 CSV 和可阅读报告，不手工抄录实验数值。"""
import csv
import json
from pathlib import Path
import statistics as st
import numpy as np

OUT = Path(__file__).resolve().parent / "results/baseline"
cases = [("breast_cancer", 0), ("banknote", 0), ("spambase", 0), ("magic", 500)]
lines = ["# 基础 SVM 与官方 LIBSVM 初步对照", "",
         "## 实验协议", "",
         "固定 C=1，RBF gamma=1/特征数；两个实现的停止容差参数均为 0.001，另用同一公式核验最终 KKT 残差。",
         "分层随机划分 75% 训练、25% 测试，seed=42；标准化只拟合实际训练子集。MAGIC 从训练池再分层抽取 500 个样本，测试集保持不变。",
         "没有参数搜索，测试集只作评估。Breast Cancer 划分沿用开发验证，因此属于开发阶段对照，不构成新的独立泛化验证。",
         "每个配置计划重复 3 次；同一划分上的重复只度量计时波动，不是泛化不确定性。训练/预测分别计时，报告均值±样本标准差。",
         "每次运行使用新进程，NumPy/OMP/MKL 线程数设为 1；训练时间包括核矩阵构建、基础版每轮诊断以及 LIBSVM 接口数据转换。",
         "数据加载、标准化、训练后核验和模型保存不计入训练时间。预测时间包含官方接口转换与内部评估，因而是现有接口级耗时。",
         "LIBSVM 开启 Shrinking，缓存 100 MB，不训练概率模型。基础版保存完整核矩阵；不声称内存预算相同。",
         "每进程预算 120 秒（包括加载、训练、预测、核验、保存）；超时后跳过同配置剩余重复，无法将其直接解释为训练耗时下界。",
         "峰值内存本轮未测量。基础模型为原始 svm.py，未作性能优化。", "",
         "## 分类效果与训练时间", "",
         "Precision、Recall、F1 均按 +1 类计算：Breast Cancer 为恶性，Banknote 为原始标签 1，Spambase 为垃圾邮件，MAGIC 为 gamma。", "",
         "|数据集|核|实现|完成次数|Accuracy|Precision|Recall|F1|训练秒（均值±SD）|预测秒（均值±SD）|SV|状态|",
         "|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---|"]
rows, diagnostics, comparisons = [], [], []
for name, limit in cases:
    for kernel in ("linear", "rbf"):
        pair = {}
        for model in ("baseline", "libsvm"):
            files = sorted(OUT.glob(f"{name}_{limit}_{kernel}_{model}_*.json"))
            records = [json.loads(f.read_text(encoding="utf-8")) for f in files]
            good = [r for r in records if "accuracy" in r]
            status = ",".join(sorted({r["status"] for r in records})) or "未运行"
            base = dict(dataset=name, limit=limit, kernel=kernel, model=model, repeats=len(good), status=status)
            if not good:
                rows.append(base)
                lines.append(f"|{name}|{kernel}|{model}|0|—|—|—|—|—|—|—|{status}|")
                continue
            r = good[0]
            pair[model] = good
            for key in ("accuracy", "precision", "recall", "f1", "n_train", "n_test", "support_vectors", "kkt_max", "gap", "equality_error", "box_error"):
                base[key] = r[key]
            for key in ("train_seconds", "predict_seconds"):
                base[key + "_mean"] = st.mean(x[key] for x in good)
                base[key + "_sd"] = st.stdev(x[key] for x in good) if len(good) > 1 else 0.
            rows.append(base)
            lines.append(f"|{name}|{kernel}|{model}|{len(good)}|{r['accuracy']:.4f}|{r['precision']:.4f}|{r['recall']:.4f}|{r['f1']:.4f}|{base['train_seconds_mean']:.5f} ± {base['train_seconds_sd']:.5f}|{base['predict_seconds_mean']:.5f} ± {base['predict_seconds_sd']:.5f}|{r['support_vectors']}|{status}|")
            diagnostics.append(f"|{name}|{kernel}|{model}|{r['n_train']}/{r['n_test']}|{r['kkt_max']:.3g}|{r['equality_error']:.3g}|{r['box_error']:.3g}|{r['gap']:.3g}|{r['iterations']}|")
            if len(good) == 1:
                lines[-1] = lines[-1].replace(" ± 0.00000", "（单次）")
                base["train_seconds_sd"] = base["predict_seconds_sd"] = None
        if len(pair) == 2:
            a, b = pair["baseline"][0], pair["libsvm"][0]
            agreement = np.mean(np.array(a["prediction"]) == b["prediction"])
            ratio = st.mean(r["train_seconds"] for r in pair["baseline"]) / st.mean(r["train_seconds"] for r in pair["libsvm"])
            ratio_text = f"{ratio:.1f}" if a["status"] == "converged" and b["kkt_max"] <= .00101 else "不比较（未达到同等收敛要求）"
            comparisons.append(f"|{name}|{kernel}|{agreement:.4%}|{ratio_text}|{a['accuracy']-b['accuracy']:+.5f}|")
lines += ["", "## 收敛核验", "", "LIBSVM 的 returned 表示接口正常返回；收敛质量以重算残差为准。基础版迭代数是全样本扫描次数，不能与 LIBSVM 变量对更新次数直接比较。",
          "", "|数据集|核|实现|训练/测试数|最大 KKT 残差|等式残差|盒约束残差|原始—对偶间隙|扫描次数|",
          "|---|---|---|---|---:|---:|---:|---:|---:|"] + diagnostics
lines += ["", "## 成对比较", "", "耗时比为基础版平均训练秒 / LIBSVM 平均训练秒，值越大表示基础版越慢。只对达到收敛要求的成对结果计算耗时比；max_iter 预测仅用于诊断。近毫秒级耗时比对环境噪声敏感。",
          "", "|数据集|核|测试预测一致率|训练耗时比|Accuracy 差值（基础版−LIBSVM）|",
          "|---|---|---:|---:|---:|"] + comparisons
lines += ["", "## 解释范围", "", "结果用于回答当前实现、固定参数与固定划分下的分类效果和接口耗时差异，不代表各数据集的最优分类效果。",
          "超时配置没有可用模型及测试预测，因此不能用其他规模的结果填补其分类指标。",
          "svm.py 每检查一个样本就执行完整核矩阵乘法；一次全扫描的该部分工作随训练样本数约为立方增长，另有 Python 变量对搜索及每轮诊断。这是代码结构分析，尚未通过剖析量化各部分贡献。",
          "基础版预测使用全部训练点，而 LIBSVM 使用支持向量；接口转换与向量化方式也不同，预测耗时不能单独归因于求解器优劣。",
          "原始记录、划分索引、标准化参数及训练好的模型均保存在本目录。试跑 spambase_200 记录单独保留，不混入上述主表。"]
lines += ["", "## Spambase 小规模试跑（单次，不计入主表）", "",
          "训练池分层抽取 200 个样本，测试集仍为 1151 个样本。没有按测试表现选择样本；这组试跑仅用于检查可运行性及估算预算。",
          "", "|核|实现|Accuracy|F1|训练秒|最大 KKT 残差|",
          "|---|---|---:|---:|---:|---:|"]
for kernel in ("linear", "rbf"):
    for model in ("baseline", "libsvm"):
        path = OUT / f"spambase_200_{kernel}_{model}_0.json"
        if path.exists():
            r = json.loads(path.read_text(encoding="utf-8"))
            if "accuracy" in r:
                lines.append(f"|{kernel}|{model}|{r['accuracy']:.4f}|{r['f1']:.4f}|{r['train_seconds']:.5f}|{r['kkt_max']:.3g}|")
(OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
keys = list(dict.fromkeys(k for row in rows for k in row))
with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
print("Report and summary saved")
