# 自实现 SVM 与 LIBSVM 实验

使用 Python 和 NumPy 实现二分类软间隔 SVM，支持线性核和 RBF 核。本项目记录基础实现、两步优化、工程改进、参数搜索和训练规模实验。

## 主要结果

|实验|结果|
|---|---|
|Banknote 线性核|V0 → V1 → V2 平均训练时间为 75.74 → 7.72 → 1.02 秒，准确率均为 97.67%|
|MAGIC 特征变换与重新选参|500 个训练样本，准确率从 83.95% 提升至 85.45%，拟合时间从 3.52 秒降至 1.19 秒|
|MAGIC 规模扩展|10,000 个训练样本，V3 和 LIBSVM 平均拟合时间为 85.49 和 1.57 秒，准确率为 86.67% 和 86.69%|

上述时间来自保存的实验批次。调参收益包含特征与参数变化，MAGIC 特征变换配置尚缺同条件 LIBSVM 对照。各次重复使用相同划分，反映计时波动。开发期间已经查看过测试集，分类改善需要在新划分上进一步验证。

## 报告入口

- [按实验框架整理的 Markdown 报告](report/SVM课程实验报告（按实验框架整理）.md)
- [各版本与 LIBSVM 详细对照](report/自实现SVM与LIBSVM详细对照.md)
- [V3 与参数搜索补充实验](report/tricks/V3与调参补充实验.md)
- [MAGIC 规模实验表](results/scale_magic/table.md)

公开包包含 Markdown 报告。后续 Word 文风修改未全部同步回 Markdown。

## 版本

|版本|文件与类|主要变化|
|---|---|---|
|V0|svm.py / SVM|基础 SMO，每次检查样本重新计算误差|
|V1|svm_optimized.py / SVMV1|误差缓存、增量更新、每轮校准|
|V2|svm_optimized.py / SVMV2|优先尝试误差差值较大的第二变量|
|V3|svm_fast.py / FastSVM|复用训练核矩阵诊断、压缩预测计算|

标签使用 −1/+1。训练后检查 stop_reason_ 和 diagnostics()，区分 converged、stalled、max_iter。

## Windows 环境与快速检查

实验使用 Python 3.11。下载脚本使用 curl.exe，超时管理使用 taskkill，峰值内存读取使用 Windows API。建议在 Windows PowerShell 中运行。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt -r requirements-figures.txt
.\.venv\Scripts\python prepare_data.py
.\.venv\Scripts\python check_svm.py
.\.venv\Scripts\python check_optimized.py
.\.venv\Scripts\python check_fast.py
```

若只需安装直接依赖，可改为安装 requirements.txt。锁定文件记录原实验环境的版本。

## 复现实验

以下命令在仓库根目录执行。

```powershell
# 基础版与官方 LIBSVM，然后运行 V1/V2
.\.venv\Scripts\python benchmark.py --timeout 120 --resume
.\.venv\Scripts\python benchmark.py --optimize --timeout 120 --resume

# V3 同期对照和训练池内部三折参数搜索
.\.venv\Scripts\python run_tricks.py
.\.venv\Scripts\python tolerance_trick.py
.\.venv\Scripts\python magic_feature_trial.py

# 固定测试集扩大训练规模
.\.venv\Scripts\python magic_scale.py
.\.venv\Scripts\python magic_scale.py --baseline
```

包内包含历史逐次 JSON 和训练测试划分。benchmark.py --resume、run_tricks.py 和 magic_scale.py 会复用已有记录。完整重新测量时，请先将 results 文件夹移到仓库外备份，再按上述顺序运行。较大规模训练需要较多时间和内存，V0 会触及预算。120 秒为整个工作进程的预算。

## 重建图表

已有结果可以直接用于生成图表。数据下载和训练不是绘图的必需步骤。

```powershell
.\.venv\Scripts\python build_course_artifacts.py
.\.venv\Scripts\python build_analysis_figures.py
.\.venv\Scripts\python summarize_tricks.py
.\.venv\Scripts\python summarize_magic_scale.py
```

write_polished_report.py 生成历史综合稿。人工编辑过的 Word 版本不由该脚本生成。

## 数据与记录

prepare_data.py 从 UCI 下载 Breast Cancer Wisconsin Diagnostic、Banknote Authentication、Spambase 和 MAGIC Gamma Telescope。来源链接与下载校验值由脚本保存到 data/manifest.json。原始数据不随本包分发，使用时请遵守各数据集来源页面的使用和引用要求。

results/baseline 和 results/optimization 保存主实验记录，results/tricks 保存调参与工程优化记录，results/scale_magic 保存规模实验记录。NPZ 包含复现实验所需的处理后划分及索引。模型权重文件可通过重新训练生成。

## 实现范围

自实现保存完整核矩阵，空间随训练样本数平方增长。运行时间取决于数据、参数和收敛状态。本项目用于课程学习和算法验证。基础实现、测试及报告在大模型辅助下完成，偏置处理和零决策值标签约定通过边界案例核验。

本包没有指定代码许可证。若需要授权他人使用、修改和分发，请由作者选择适当的 LICENSE。
