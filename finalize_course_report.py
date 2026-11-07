"""将实测表和结果段落写入报告初稿；保留人工撰写的解释与反思。"""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
out = root / "report"
rows = json.loads((out / "summary.json").read_text(encoding="utf-8"))
assert not any(r["status"] == "missing" for r in rows)
def get(dataset, kernel, model):
    return next(r for r in rows if (r["dataset"],r["kernel"],r["model"]) == (dataset,kernel,model))
spam = get("spambase","rbf","v2")
magic = get("magic","linear","v2")
m1 = get("magic","rbf","v1")
m2 = get("magic","rbf","v2")
text = f'''结果还揭示了两个不能被平均加速比掩盖的现象。Spambase 线性核中，V0、V1、V2 都触及进程预算，因此两项优化没有使这一配置在当前预算内完成。对 Spambase RBF，V0 和 V1 超时，V2 则在平均 {spam['time']:.2f} 秒、{spam['iterations']} 轮内收敛，Accuracy 为 {spam['accuracy']:.2%}，与 LIBSVM 相同。变量选择的作用因此与具体配置有关，不能说所有较大数据都已解决。

MAGIC 线性核中，V0 和 V1 都达到 2,000 轮上限，V1 虽然运行更快，却未解决停止精度问题；V2 在 {magic['iterations']} 轮、平均 {magic['time']:.2f} 秒内收敛，Accuracy 为 {magic['accuracy']:.2%}。MAGIC RBF 的 V1、V2 则分别为 {m1['time']:.2f} 秒/{m1['iterations']} 轮与 {m2['time']:.2f} 秒/{m2['iterations']} 轮。这里，“减少每轮成本”和“减少迭代数量”的区别比单一耗时数字更清楚。

本阶段每轮校准前记录的最大误差缓存漂移约为 2.14×10⁻¹³，成功收敛的配置均通过最终独立残差检查。它支持缓存实现的数值一致性，但不要求各版本的迭代轨迹完全一致。
'''
p = out / "实验报告初稿.md"
s = p.read_text(encoding="utf-8")
s = s.replace("<!-- PERFORMANCE_TABLE -->", (out / "performance_table.md").read_text(encoding="utf-8"))
s = s.replace("<!-- FINAL_FINDINGS -->", text)
p.write_text(s,encoding="utf-8")
print(p)
