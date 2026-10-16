# GEARS：遗传扰动表达预测

用户选择 GEARS 后，按本页完成资源、输入、训练/加载、预测和结果处理。GEARS 的 KO/OE 含义来自训练实验：KO/CRISPRi 数据训练的模型用于相应下调扰动，CRISPRa/OE 数据训练的模型用于相应上调扰动。`predict` 接收基因列表，没有把同一权重切换为 KO 或 OE 的参数。

## 1. 获取资源

- [官方仓库与安装](https://github.com/snap-stanford/GEARS)：先在所选环境安装与 torch 匹配的 PyG，再安装 `cell-gears` 或固定 commit 的源码。
- [数据教程](https://github.com/snap-stanford/GEARS/blob/master/demo/data_tutorial.ipynb)：`PertData` 可获取内置扰动数据及图资源，也可处理用户自己的 AnnData。内置数据仅在用户选定后使用。
- 已有模型用 `save_model` 导出的目录，至少含 `config.pkl` 和 `model.pt`；一起取得训练数据的基因顺序、扰动词表、图资源、split 和预处理记录。官方 README 的 Colab 提供 Norman 训练模型示例。
- 没有匹配模型时，从用户选定的扰动训练数据拟合并保存 checkpoint。对照表达矩阵与扰动训练数据是不同的数据角色。

## 2. 准备输入

1. 单次模型使用匹配的细胞背景。输入包含 ctrl 与多种真实扰动，每个条件有多个细胞。混合细胞背景时先选择本次背景。
2. `.var['gene_name']` 是唯一 gene symbol；`.obs['condition']` 使用 `ctrl`、`GENE+ctrl` 或 `GENE_A+GENE_B`；`.obs['cell_type']` 保留实际背景。gRNA 标签先映射到靶基因，保存映射表。
3. 自建数据从原始计数副本进行与训练方案一致的 QC、总量归一化和 log1p；例如采用每细胞总量 1e4 后 log1p 时，记录该尺度。已处理数据先核实尺度，避免重复归一化。以 CSR 稀疏矩阵交给 `new_data_process`；该函数构图和计算 DE，不替用户决定原始计数预处理。
4. 留存原始计数、细胞/基因筛选记录。按扰动条件而非随机拆细胞划分训练/验证/测试。单基因路线用 `simulation_single`，含组合路线可用 `simulation`；复用 checkpoint 时复用原 split。
5. 联合预测需要组合扰动训练支持；目标要在 `model.pert_list` 图节点中。分别登记图覆盖、训练见过与否、测试条件，不能仅以“基因在表达矩阵中”作为覆盖判定。

## 3. 调用示例

下面函数接收已经处理且核实尺度的 AnnData。将其保存到本次分析目录的代码文件，调用参数取自本次启动选择。`targets=[['STAT3'], ['JUN']]` 表示分别预测，`[['STAT3', 'JUN']]` 表示联合预测；基因名由用户选择。

```python
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from gears import PertData, GEARS

def run_gears(adata, targets, output_dir, *, task, split, checkpoint=None,
              epochs=20, device="cpu", seed=1):
    if task not in {"ko", "oe"}:
        raise ValueError("task 必须与训练实验的 KO/OE 语义一致")
    if not targets or any(not group for group in targets):
        raise ValueError("请提供用户选定的靶基因")
    if not {"condition", "cell_type"}.issubset(adata.obs):
        raise ValueError("缺 condition/cell_type")
    genes = adata.var["gene_name"].astype(str).tolist()
    if len(set(genes)) != len(genes) or adata.obs.cell_type.nunique() != 1:
        raise ValueError("需要唯一基因名和单一细胞背景")
    if not (adata.obs.condition == "ctrl").any():
        raise ValueError("缺少 ctrl 细胞")
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(values).all():
        raise ValueError("表达矩阵含非有限值")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    prepared = adata.copy()
    prepared.X = sparse.csr_matrix(prepared.X)
    pdata = PertData(str(out / "data"))
    pdata.new_data_process(dataset_name="user_data", adata=prepared)
    pdata.prepare_split(split=split, seed=seed)
    pdata.get_dataloader(batch_size=32, test_batch_size=128)
    model = GEARS(pdata, device=device, weight_bias_track=False)
    if checkpoint is None:
        model.model_initialize(hidden_size=64)
        model.train(epochs=epochs)
        model.save_model(str(out / "checkpoint"))
    else:
        model.load_pretrained(str(checkpoint))
    genes = pdata.adata.var.gene_name.astype(str).tolist()
    baseline = np.asarray(pdata.adata[pdata.adata.obs.condition == "ctrl"].X.mean(axis=0)).ravel()
    rows, predictions, uncertainties, errors = [], [], [], []
    for index, group in enumerate(targets):
        try:
            result = model.predict([group])
            native, uncertainty = result if isinstance(result, tuple) else (result, {})
            key = "_".join(group)
            pred = np.asarray(native[key]).reshape(-1)
            if pred.shape != baseline.shape or not np.isfinite(pred).all():
                raise ValueError("输出轴或数值无效")
            rows.append({"condition_id": f"target_{index:04d}", "task": task,
                         "targets": json.dumps(group), "native_key": key})
            predictions.append(pred)
            if key in uncertainty:
                uncertainties.append({"condition_id": rows[-1]["condition_id"],
                                      "native_uncertainty": float(uncertainty[key])})
        except Exception as exc:
            errors.append({"targets": group, "error": str(exc)})
    (out / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    if not rows:
        raise RuntimeError("所有目标均未完成，见 errors.json")
    prediction = np.stack(predictions)
    response = prediction - baseline
    metadata = pd.DataFrame(rows)
    metadata.to_csv(out / "conditions.csv", index=False)
    np.savez_compressed(out / "predictions.npz", prediction=prediction, baseline=baseline,
                        response=response, genes=np.asarray(genes, dtype=str),
                        condition_id=metadata.condition_id.to_numpy(dtype=str))
    if uncertainties:
        pd.DataFrame(uncertainties).to_csv(out / "native_uncertainty.csv", index=False)
    pd.DataFrame(response, index=metadata.condition_id, columns=genes).to_csv(out / "response.csv")
    return out
```

该片段展示模型调用和全量保存；执行者按 [执行手册](execution-playbook.md) 补齐运行 manifest、资源哈希、输入尺度、split、报告和图形。加载旧 checkpoint 前先对照其训练基因顺序/图节点，不能只依靠张量维度相同判断兼容。

## 4. 整理结果

- `predict` 的主要输出是每个扰动的平均预测表达，键由目标列表以 `_` 拼接。保留独立条件 ID 和原始目标列表，避免通过拆字符串恢复联合条件。
- 本例响应 = 平均预测 − 数据中 ctrl 均值，单位沿用训练表达尺度。官方当前预测会随机抽取 300 个 ctrl；记录种子与抽样方式。这里的总体 ctrl 均值用于背景比较，不能当作逐细胞配对真值。
- 表格交付全基因预测/响应、目标覆盖、条件和失败项；按响应绝对值展示 Top 基因、热图及适用富集。KO/OE 在标题中写明实验语义与训练来源。
- 有留出实测扰动时按相同基因轴计算表达与 delta 的 Pearson、MSE、方向一致率，并分开列全基因和预先确定的 DE 基因结果。保存 split，保证测试条件未参与训练。
- `uncertainty=True` 的附加返回值按该版本源码定义保存为原生不确定性量，不能直接标为置信区间。

API 依据：[PertData](https://github.com/snap-stanford/GEARS/blob/master/gears/pertdata.py)、[GEARS 模型](https://github.com/snap-stanford/GEARS/blob/master/gears/gears.py)、[预测抽样](https://github.com/snap-stanford/GEARS/blob/master/gears/utils.py)，核对日期 2026-10-16。
