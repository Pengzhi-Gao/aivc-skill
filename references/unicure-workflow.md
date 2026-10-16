# UniCure：细胞与药物表征到表达响应

本页以官方 LINCS 978 基因模型为示例：细胞基线 → 匹配 UCE-LoRA 表征，SMILES → Uni-Mol 表征，再与剂量一起预测处理后表达。SciPlex/PTC 分支使用各自的模型类、基因轴与处理方式，按官方对应示例配置。

## 1. 获取资源

[官方仓库](https://github.com/ZexiChen502/UniCure)的 README 提供 `data`、`requirement`、`result` 下载入口，以及 [Figshare 32077296](https://doi.org/10.6084/m9.figshare.32077296) 归档。需要：

- 所选 LINCS checkpoint，例如 `result/11/lincs2020/Unicure_best_model.pth`。
- LINCS 基因列与顺序、表达预处理记录、细胞 ID 和对应 control 数据。
- `lincs2020_uce_lora_emb.parquet` 与 `lincs2020_unimol_emb.parquet`，或按下述步骤新生成的兼容表征。
- 为新样本生成细胞表征时，另配 `requirement/UCE_pretraining_files/` 的预训练文件、protein embeddings、token 与染色体/基因位置资源。文件名单和下载入口见官方 README；配置路径须与 `preprocessing.py`、`utils.py` 实际读取位置一致。

按官方 `environment.yml` 或 requirements 配置环境；Uni-Mol 可用独立编码环境。推理前核对 UniCure、UCE 和 Uni-Mol 的版本与表征维度。既有表征可以直接进入响应预测步骤。

## 2. 处理细胞与药物输入

### 细胞基线

用户选择现成 LINCS 背景时，用相同 `new_cid` 连接 cell embedding 与 control；保留所有对应对照及其 gene order。用户提供自己的数据时：

1. 建立 cells/samples × genes 的 control parquet。示例第一列为 `sample`，后面是 gene symbol 表达列；名称映射与训练版本一致，保留原始矩阵和转换记录。
2. 按 `preprocessing.py` 的选定数据分支准备表达。该源码在生成句子时会处理负值并按表达权重抽样；保存实际变换，原始负值、缺失与观测零值分开记录。用于计算响应的 baseline 必须在输出 checkpoint 的 978 基因表达尺度上，不能直接拿 UCE 的 token 权重作基线。
3. 将官方源码目录作为工作目录，在配置好的环境中调用下列函数。函数输出文件使用本次分析目录的绝对路径；目标文件应为新路径，官方函数会跳过已存在文件。

```python
from preprocessing import generate_esm2_emb
from utils import uce_emb

def prepare_cells(control_path, token_path, embedding_path, checkpoint):
    generate_esm2_emb(control_path=control_path, gene_columns_start=1,
                      save_dir=token_path, dataset_name="clinical")
    uce_emb(esm2_emb_df_path=token_path, esm2_control_df_path=control_path,
            model_path=checkpoint, uce_emb_df_path=embedding_path,
            index_name="sample")
```

`generate_esm2_emb` 在此生成 UCE 输入 token/掩码；后一步输出 1280 维细胞表征。保存细胞行序及 `sample` 对照，固定抽样种子。使用预计算表征时核实生成 checkpoint 与待推理权重对应。

### 药物和剂量

- 药物库包含稳定 `drug` ID、名称、canonical SMILES 与 `dose_uM`。先匹配官方 Uni-Mol 表征中的索引；新结构按官方所用 Uni-Mol 版本及构象/CLS 设置生成，不能以其他编码器同维度向量替代。
- 从同一批结构中抽取已有药物做编码一致性检查，记录构象种子和表征来源。编码失败的药物单列为 `no_prediction`。
- LINCS 示例读取 512 维 Uni-Mol 表征；拼接 `log10(dose_uM + 1)` 并用同一剂量值补齐至 528 维。细胞与药物 batch 行数对齐。剂量单位来自配套训练表，外部单位先换算。

新分子的编码入口可用 [Uni-Mol 官方 UniMolRepr](https://github.com/deepmodeling/Uni-Mol/blob/main/unimol_tools/unimol_tools/predictor.py)。下面的 `remove_hs` 取所选训练表征的设置，在已有药物对照通过后生成新药表征；记录实际 Uni-Mol checkpoint 和构象设置。

```python
import numpy as np
import pandas as pd
from unimol_tools import UniMolRepr

def encode_drugs(smiles_by_id, *, remove_hs):
    ids = list(smiles_by_id)
    if not ids:
        raise ValueError("药物列表为空")
    encoder = UniMolRepr(data_type="molecule", model_name="unimolv1", remove_hs=remove_hs)
    result = encoder.get_repr([smiles_by_id[k] for k in ids], return_atomic_reprs=False)
    embeddings = np.asarray(result["cls_repr"], dtype=np.float32)
    if embeddings.shape != (len(ids), 512) or not np.isfinite(embeddings).all():
        raise ValueError("Uni-Mol 输出维度或数值不匹配")
    return pd.DataFrame(embeddings, index=ids, columns=[str(i) for i in range(512)])
```

## 3. 响应预测示例

以下函数直接使用官方 `UniCure` 的 `pertrub_forward`，输入为已生成的表征。它仅加载这一前向路径所需参数，适用于已核对的 LINCS 模型结构；细胞表征生成仍使用上一步的完整 UCE 路线。

```python
import numpy as np
import torch
from model import UniCure

def predict_unicure(checkpoint, cell_embeddings, drug_embedding, dose_uM,
                    genes, baseline_expression, device="cpu", batch_size=64):
    cells = np.asarray(cell_embeddings, dtype=np.float32)
    drug = np.asarray(drug_embedding, dtype=np.float32).reshape(-1)
    baseline = np.asarray(baseline_expression, dtype=np.float32)
    if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] != 1280:
        raise ValueError("LINCS 示例需要 N × 1280 细胞表征")
    if drug.shape != (512,) or len(genes) != 978 or len(set(genes)) != len(genes):
        raise ValueError("需要 512 维药物表征和有序的 978 个唯一基因")
    if baseline.ndim != 2 or baseline.shape[0] == 0 or baseline.shape[1] != len(genes):
        raise ValueError("baseline 必须是与输出同尺度同基因序的对照矩阵")
    if not all(np.isfinite(x).all() for x in (cells, drug, baseline)):
        raise ValueError("输入含非有限值")
    if not np.isfinite(dose_uM) or dose_uM < 0 or batch_size < 1:
        raise ValueError("检查剂量和 batch_size")
    model = UniCure(output_size=len(genes))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # 此路径不调用 uce_lora；其余参数完整、严格加载。
    forward_state = {k: v for k, v in state.items() if not k.startswith("uce_lora.")}
    model.load_state_dict(forward_state, strict=True)
    model.to(device).eval()
    condition = np.concatenate([drug, np.full(16, np.log10(dose_uM + 1), dtype=np.float32)])
    parts = []
    with torch.no_grad():
        for start in range(0, len(cells), batch_size):
            cell_batch = torch.as_tensor(cells[start:start + batch_size], device=device)
            drug_batch = torch.as_tensor(condition, device=device).repeat(len(cell_batch), 1)
            parts.append(model("pertrub_forward", cell_batch, drug_batch).cpu().numpy())
    per_cell = np.concatenate(parts)
    if per_cell.shape != (len(cells), len(genes)) or not np.isfinite(per_cell).all():
        raise ValueError("预测输出轴或数值无效")
    prediction = per_cell.mean(axis=0)
    baseline_mean = baseline.mean(axis=0)
    return per_cell, prediction, baseline_mean, prediction - baseline_mean
```

按 `(sample/cell_background, drug, dose, time)` 逐条件调用。时间仅在模型实际支持时作为预测维度，否则记为背景元数据。可用已有嵌入减少响应预测的内存开销；新样本的完整表征生成另行核算资源。其他 checkpoint 结构需改用对应官方 loader，并检查所有缺失/多余参数。

## 4. 结果接到药筛交付

1. 保存 `per_cell`、对应细胞 ID、完整 `prediction`、`baseline_mean`、`response` 与 gene order。若表征按细胞背景聚合，记录聚合粒度。
2. 将逐条件返回向量堆成 conditions × genes。使用 [common.py](../examples/lung_cancer/drug_screen/common.py) 的 `RunDir.save_predictions` 保存，再用 `condition_quality`、`drug_result_tables` / `RunDir.finish` 生成恢复分类和逐药逐条件 Top 30。
3. 疾病方向评分按 [逆转评分规则](downstream-results.md) 对齐参考、基因和尺度；缺评分参照时保留响应及 Top 30，恢复类别记为无法判定。输出有序基因列不能直接从任意 control 文件推断，必须与 checkpoint 的训练目标列核对。
4. 对每个细胞背景比较药物与剂量间的响应差异；有实测处理样本时另算表达及 delta 的相关/误差。输入表征、响应预测和独立验证分别记录状态。

代码依据：[quick_pred.py](https://github.com/ZexiChen502/UniCure/blob/main/quick_pred.py)、[model.py](https://github.com/ZexiChen502/UniCure/blob/main/model.py)、[preprocessing.py](https://github.com/ZexiChen502/UniCure/blob/main/preprocessing.py)、[utils.py](https://github.com/ZexiChen502/UniCure/blob/main/utils.py)，核对日期 2026-10-16。
