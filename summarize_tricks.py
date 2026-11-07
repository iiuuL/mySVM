"""汇总 V3 与调参试验；严格保留未收敛和超时，不按测试分数选择配置。"""
from pathlib import Path
import json
import statistics as st
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SRC=ROOT/"results/tricks"
OUT=ROOT/"report/tricks"
OUT.mkdir(exist_ok=True)
names=["breast_cancer","banknote","spambase","magic"]
titles=["Breast Cancer","Banknote","Spambase","MAGIC 500"]
rows=[]
for name in names:
    for kernel in ("linear","rbf"):
        for setting in (["default"] if kernel=="linear" else ["default","best","economical"]):
            for model in (["v2","v3","libsvm"] if setting=="default" else ["v3","libsvm"]):
                files=sorted(SRC.glob(f"{name}_{kernel}_{setting}_{model}_*.json"))
                records=[json.loads(f.read_text(encoding="utf-8")) for f in files]
                good=[r for r in records if "accuracy" in r]
                row=dict(dataset=name,kernel=kernel,setting=setting,model=model,status=records[0]["status"] if records else "missing",repeats=len(good))
                if good:
                    r=good[0]
                    for key in ("accuracy","f1","C","gamma","iterations","kkt_max","gap","sv"):
                        row[key]=r[key]
                    row["train_seconds"]=st.mean(x["train_seconds"] for x in good)
                    row["train_sd"]=st.stdev(x["train_seconds"] for x in good) if len(good)>1 else None
                    row["predict_ms"]=1000*st.mean(x["predict_seconds"] for x in good)
                    row["max_prediction_error"]=max((x["prediction_error"] or 0) for x in good)
                    assert all(x["prediction"]==r["prediction"] for x in good)
                    feasible=all(x["kkt_max"]<=.00101 and x["equality_error"]<1e-6 and x["box_error"]<1e-7 for x in good)
                    if row["status"]=="returned" and not feasible:
                        row["status"]="returned_residual_above_threshold"
                    if row["status"]=="converged":
                        assert feasible, (name,kernel,setting,model)
                    if len(good)!=len(records):
                        row["status"]="partial_"+next(x["status"] for x in records if "accuracy" not in x)
                rows.append(row)
def get(name,kernel,setting,model):
    return next(r for r in rows if (r['dataset'],r['kernel'],r['setting'],r['model'])==(name,kernel,setting,model))
def valid(r):return r['status'] in ('converged','returned')
(OUT/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
def save(fig,name):
    for ext in ('png','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for ax,name,title in zip(axes.flat,names,titles):
    search=json.loads((SRC/f'{name}_selection.json').read_text(encoding='utf-8'))
    values=np.array([r['cv_accuracy']*100 for r in search['records']]).reshape(4,5)
    ax.imshow(values,cmap='YlGnBu',vmin=60,vmax=100,aspect='auto')
    ax.set_title(title);ax.set_xticks(range(5),['0.01/d','0.1/d','1/d','10/d','100/d']);ax.set_yticks(range(4),['0.1','1','10','100']);ax.set_xlabel('gamma');ax.set_ylabel('C')
    for i in range(4):
        for j in range(5):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]>87 else 'black')
fig.suptitle('训练池内部三折验证准确率（%）；测试集未参与参数选择')
save(fig,'validation_grid')

fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
labels=[f'{t}\n{k}' for t in titles for k in ('linear','RBF')]
for j,model in enumerate(('v2','v3')):
    for i,(name,kernel) in enumerate((n,k) for n in names for k in ('linear','rbf')):
        r=get(name,kernel,'default',model)
        for ax,key in zip(axes,['train_seconds','predict_ms']):
            if valid(r):ax.bar(i+(j-.5)*.32,r[key],.3,color=['#8696a7','#3795b0'][j],label=model.upper() if i==0 else None)
            else:ax.text(i+(j-.5)*.32,.95,'超时' if r['status']=='timeout' else r['status'],transform=ax.get_xaxis_transform(),rotation=90,ha='center',va='top',fontsize=8)
for ax,title in zip(axes,['训练秒（同参数、同容差）','预测毫秒（预热后七次中位数，再跨训练平均）']):
    ax.set_xticks(range(8),labels,fontsize=8);ax.set_yscale('log');ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
save(fig,'fixed_parameter_speed')

fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for j,setting in enumerate(('default','best','economical')):
    for i,name in enumerate(names):
        r=get(name,'rbf',setting,'v3')
        for ax,key in zip(axes,('accuracy','train_seconds')):
            if valid(r):
                value=r[key]*(100 if key=='accuracy' else 1)
                ax.scatter(i+(j-1)*.18,value,s=42,color=['#8696a7','#e59a3b','#3795b0'][j],label=['默认','验证最高','近优较小 C'][j] if i==0 else None)
            else:ax.text(i+(j-1)*.18,.94,'未达标',transform=ax.get_xaxis_transform(),rotation=90,ha='center',va='top',fontsize=8)
for ax in axes:ax.set_xticks(range(4),titles);ax.grid(axis='y',alpha=.2);ax.legend()
axes[0].set_ylabel('测试 Accuracy / %');axes[0].set_title('调参能否提高测试准确率？')
axes[1].set_yscale('log');axes[1].set_ylabel('训练秒');axes[1].set_title('更高验证准确率可能伴随更长训练')
save(fig,'tuning_tradeoff')

lines=['# V3 工程优化与参数搜索补充实验','',
'本轮保留原 V0–V2，新增 V3。V3 每轮诊断复用已构建的核矩阵，仍每轮检查同一残差；训练结束独立重算核验。线性核预测预计算 w=Xᵀ(α⊙y)，RBF 仅保留严格非零系数（不以阈值剪裁），并检查与完整预测路径的差异。',
'','## 协议与参数选择','',
'固定参数实验同期运行 V2、V3、官方 LIBSVM，交替顺序，三次训练重复；线程设置为 1，C=1、gamma=1/d、tol=0.001，最大 2,000 轮、每进程 120 秒。计时包括训练内诊断和模型预测准备。预测计时预热一次、重复七次取中位数，再跨训练取平均，不能与旧报告单次预测耗时直接混比。',
'','调参使用原训练池内部三折分层验证（seed=2026），每折单独拟合标准化，测试集及原训练样本范围不变。搜索 C={0.1,1,10,100}，gamma={0.01,0.1,1,10,100}/d，共每数据集 20 组、60 次 LIBSVM 验证训练；四个数据集合计 240 次。用 LIBSVM 选参数后，将相同参数用于 V3 与 LIBSVM，不能把调参提高的准确率归因于 V3 代码本身。',
'','“best”最大化验证 Accuracy，完全并列时选较小 C，再较小 gamma。“economical”在验证最高准确率 0.5 个百分点内优先选较小 C，再较小 gamma；这只是预设的参数偏好，不保证训练更快。全部选择在测试评估前冻结，未根据测试分数回头修改搜索范围。MAGIC 只使用原先的 500 个训练样本。既有测试集已在早期开发查看，本次是探索性补充，不是新的盲测。',
'','![验证网格](validation_grid.png)','',
'|数据|默认验证 Accuracy|最高验证 Accuracy|best C|best gamma|搜索总秒|',
'|---|---:|---:|---:|---:|---:|']
for name in names:
    s=json.loads((SRC/f'{name}_selection.json').read_text(encoding='utf-8'));b=s['best']
    lines.append(f"|{name}|{s['default']['cv_accuracy']:.2%}|{b['cv_accuracy']:.2%}|{b['C']:g}|{b['gamma']:.6g}|{s['search_seconds']:.2f}|")
lines+=['','## 代码里还有哪里在重复工作？','',
'读 V2 的扫描循环时，我注意到误差虽然已经缓存，停止检查仍会调用训练集上的预测函数。预测函数按通用输入处理，又计算了一次训练集与自身的核矩阵。扫描开始前已经得到同一矩阵 K，停止条件只需要本轮的决策值。两版对应的代码是：','',
'```python','# V2：诊断内部调用 decision_function(self.X_)','self.diagnostics_ = self.diagnostics()','# V3：使用本次训练已经构造的 K','self.diagnostics_ = self.diagnostics(K @ (a * y) + self.b_)','```','',
'这一行仍按当前乘子做矩阵向量乘法，因此保留了原来的停止检查。预测另有一处可省：对线性核先保存 w，预测一批样本时只做 X @ w + b；对 RBF 核只计算乘子非零的训练样本与测试样本之间的核值。为了检查压缩有没有改变边界判断，每次训练后都额外计算完整决策值，逐个核对测试标签。独立核验在训练计时结束后执行。','',
'## 同参数速度对照','', '仅对两端都收敛的结果计算加速比，超时没有可用耗时。三次训练均值用于下表。', '',
'|数据／核|V2训练秒|V3训练秒|训练加速比|V2预测ms|V3预测ms|预测加速比|测试预测一致率|',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for name in names:
    for kernel in ('linear','rbf'):
        a,b=get(name,kernel,'default','v2'),get(name,kernel,'default','v3')
        if valid(a) and valid(b):
            x=json.loads((SRC/f'{name}_{kernel}_default_v2_0.json').read_text());y=json.loads((SRC/f'{name}_{kernel}_default_v3_0.json').read_text())
            agree=np.mean(np.array(x['prediction'])==y['prediction'])
            lines.append(f"|{name}/{kernel}|{a['train_seconds']:.4f}|{b['train_seconds']:.4f}|{a['train_seconds']/b['train_seconds']:.2f}×|{a['predict_ms']:.4f}|{b['predict_ms']:.4f}|{a['predict_ms']/b['predict_ms']:.2f}×|{agree:.2%}|")
        else:lines.append(f"|{name}/{kernel}|{a['status']}|{b['status']}|—|—|—|—|—|")
lines+=['','![固定参数速度](fixed_parameter_speed.png)','', '## 调参结果：展示所有选择，避免只挑测试集最优者','',
'|数据|选择|实现|C|gamma|测试 Accuracy|F1|训练秒±SD|KKT残差|状态|',
'|---|---|---|---:|---:|---:|---:|---|---:|---|']
for name in names:
    for setting in ('default','best','economical'):
        for model in ('v3','libsvm'):
            r=get(name,'rbf',setting,model)
            if 'accuracy' in r:
                sd=f" ± {r['train_sd']:.4f}" if r['train_sd'] is not None else '（单次）'
                lines.append(f"|{name}|{setting}|{model}|{r['C']:g}|{r['gamma']:.6g}|{r['accuracy']:.2%}|{r['f1']:.4f}|{r['train_seconds']:.4f}{sd}|{r['kkt_max']:.3g}|{r['status']}|")
            else:lines.append(f"|{name}|{setting}|{model}|—|—|—|—|—|—|{r['status']}|")
lines+=['','![调参权衡](tuning_tradeoff.png)','', '## 结果解释','',
'以上默认参数加速与调参收益应分开解释：前者保持优化问题不变，后者改变了模型的超参数。验证集更好不保证测试集更好；较小 C 也不保证更短训练。对超时或达到扫描上限的配置，不把预测指标作为充分收敛的性能结果，也不计算公平加速比。',
'','LIBSVM 搜索耗时在参数表单列，不计入最终单模型训练时间；实际首次获得调参模型的成本还包含这部分搜索。所有原始选择、折索引、逐次记录与模型位于 results/tricks，源代码和协议哈希见 protocol.json。',
'','复现：`.\\.venv\\Scripts\\python check_fast.py`；`.\\.venv\\Scripts\\python run_tricks.py`；`python summarize_tricks.py`。运行脚本自动复用已存在记录，若更改协议请使用新结果目录。']
tolerance=json.loads((SRC/'tolerance_magic_rbf.json').read_text(encoding='utf-8'))
lines+=['','## 提前停止能省多少时间？','','在 MAGIC 的 500 个训练样本上保持 RBF 参数不变，只改变允许的最大 KKT 残差。每种容差训练三次。','',
        '|容差|扫描次数|平均训练秒|测试 Accuracy|F1|实测最大 KKT 残差|','|---:|---:|---:|---:|---:|---:|']
for tol in (1e-2,1e-3,1e-4):
    runs=[r for r in tolerance if r['tol']==tol]
    assert len(runs)==3 and all(r['status']=='converged' and r['kkt_max']<=tol for r in runs)
    r=runs[0]
    lines.append(f"|{tol:g}|{r['iterations']}|{st.mean(x['train_seconds'] for x in runs):.4f}|{r['accuracy']:.2%}|{r['f1']:.4f}|{r['kkt_max']:.3g}|")
lines+=['','这组时间只衡量不同停止要求的成本。前面的 V2 与 V3 加速比始终使用相同容差。复现本表请在汇总前运行 `.\\.venv\\Scripts\\python tolerance_trick.py`。']
lines+=['','## 回到最初的问题：能否同时更准、更快？','',
        '我先用同一组参数检查代码修改的作用，再看训练集选出的参数能否带来额外收益。下表将原 V2 默认 RBF 与验证准确率最高的 V3 比较。时间是参数确定后的一次训练均值，搜索成本见前表。','',
        '|数据|V2 默认 → V3 验证最高：测试正确数|训练秒|观察|','|---|---:|---:|---|']
for name in names:
    base=get(name,'rbf','default','v2')
    tuned=get(name,'rbf','best','v3')
    if valid(base) and valid(tuned):
        n=json.loads((SRC/f'{name}_rbf_best_v3_0.json').read_text())['n_test']
        before,after=(round(x['accuracy']*n) for x in (base,tuned))
        change='多判对' if after>before else '少判对' if after<before else '正确数持平'
        if after!=before:change+=f' {abs(after-before)} 个'
        speed='训练缩短' if tuned['train_seconds']<base['train_seconds'] else '训练变长'
        lines.append(f"|{name}|{before} → {after}（共 {n}）|{base['train_seconds']:.3f} → {tuned['train_seconds']:.3f}|{change}，{speed}|")
    else:
        lines.append(f"|{name}|—|—|V2 默认 {base['status']}，V3 验证最高 {tuned['status']}|")
lines+=['','Banknote 的默认 RBF 在这次划分上已全对。我会保留默认参数，采用 V3 省下重复构造核矩阵的时间。Breast Cancer 的提升可以具体数成一个样本，因此我只把它作为这次划分上的改善。Spambase 的线性核在两个版本中都超过进程预算，说明省掉部分重复计算还不足以解决这组训练的耗时问题。']
(OUT/'V3与调参补充实验.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Saved tricks report, tables and three figures')
