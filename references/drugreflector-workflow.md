# DrugReflector：表达变化签名到药物排名

用户选择 DrugReflector 时，输入是目标状态转换的基因签名，输出是 checkpoint 覆盖化合物的原生排名。疾病恢复任务把转换方向设为疾病 → 目标健康状态；程序从有方向的签名产生候选优先级。

## 1. 获取资源

- [官方仓库](https://github.com/Cellarity/drugreflector)提供模型、签名函数和示例；在用户指定环境按 `pyproject.toml` 安装。
- [Zenodo 16912444](https://zenodo.org/records/16912444)提供三个 fold checkpoint，下载后核对实际文件名和哈希，并按 fold 0/1/2 传入。源码归档 [16921928](https://zenodo.org/records/16921928) 与 checkpoint 归档用途不同。
- [16921905](https://zenodo.org/records/16921905)是论文示例 transition signatures；用户选择自己的数据时自行构造对应签名。

API 核对以 [DrugReflector 类](https://github.com/Cellarity/drugreflector/blob/main/drugreflector/drug_reflector.py)、[ensemble](https://github.com/Cellarity/drugreflector/blob/main/drugreflector/ensemble_model.py) 与[根目录 utils.py](https://github.com/Cellarity/drugreflector/blob/main/utils.py) 为准。2026-10-16 的包 `__init__.py` 未导出 `compute_vscores_adata`，按下例从根目录文件加载；不要与包内只处理缩放的 `drugreflector/utils.py` 混用。

## 2. 构造输入签名

1. 用户提供疾病与目标状态的表达数据/标签，或已有 v-score 表。记录物种、背景和统计单位；scRNA 比较同类细胞，保留供体/批次来源。bulk 样本用同背景独立重复。表达矩阵先按既定方法归一化并 log 转换；签名输入采用对应尺度。
2. 用真实注释将 Ensembl/其他 ID 映射成 HGNC gene symbol，并解决重复映射。上游字符串大写/去符号步骤不能替代 ID 注释映射。
3. 官方 v-score 为 `(mean_target - mean_source) / sqrt(var_source + var_target)`；两组方差之和为零时分母使用 1，方差为 `ddof=0`。疾病恢复任务令 `group1_value=疾病`、`group2_value=目标状态`。保存方向、分组和完整签名，输入已有有向签名时核实其定义后再决定是否反号。
4. 模型覆盖以三个 checkpoint 各自的基因清单核对，另存缺失基因。官方实现会给缺失基因补零；AIVC 默认先补齐必需输入。若本研究明确采用有依据的缺失基因补零方案，单独保留缺失掩码、覆盖率、方案依据与结果标记。

## 3. 调用示例

先从安装目录外按文件路径加载签名函数，避免同名 `utils` 冲突。`adata` 使用本次已归一化的表达数据；若提供 `layer`，先验证其存在，避免上游静默回退到 `.X`。

```python
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from drugreflector import DrugReflector

def make_signature(repo_dir, adata, group_col, source_state, target_state, layer=None):
    if layer is not None and layer not in adata.layers:
        raise ValueError("指定的表达层不存在")
    path = Path(repo_dir) / "utils.py"
    spec = importlib.util.spec_from_file_location("aivc_drugreflector_signature", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    signature = helper.compute_vscores_adata(
        adata, group_col=group_col, group1_value=source_state,
        group2_value=target_state, layer=layer)
    if not np.isfinite(signature.to_numpy()).all() or signature.std() == 0:
        raise ValueError("签名必须包含有效且有差异的数值")
    return signature.astype(float)

def rank_compounds(signature, checkpoints, output_dir, selected_compounds):
    # selected_compounds 是用户选定药物库映射到 checkpoint 原生 ID 后的列表。
    if not selected_compounds:
        raise ValueError("请提供用户选定的候选药物 ID")
    if not signature.index.is_unique:
        raise ValueError("请先合并重复基因映射")
    if not np.isfinite(signature.to_numpy()).all() or signature.std() == 0:
        raise ValueError("签名无效")
    model = DrugReflector(checkpoint_paths=[str(p) for p in checkpoints])
    missing = {str(i): sorted(set(names) - set(signature.index))
               for i, names in enumerate(model.model.dimensions["var_names"])}
    if any(missing.values()):
        raise ValueError(f"先补齐模型所需基因或明确缺失处理方案：{missing}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    signature.to_csv(out / "input_signature.csv", header=["vscore"])
    # 不截断模型输出；保存全部原生分类，再提取本次候选库。
    result = model.predict(signature.copy(), n_top=None)
    result.to_csv(out / "native_rankings.csv")
    transition = signature.name if signature.name is not None else "vscore"
    table = pd.DataFrame({"drug": result.index,
                          "rank": result[("rank", transition)].to_numpy(),
                          "logit": result[("logit", transition)].to_numpy(),
                          "prob": result[("prob", transition)].to_numpy()})
    if not np.isfinite(table[["rank", "logit", "prob"]].to_numpy()).all():
        raise ValueError("模型排名含非有限值，原始输出已保存")
    requested = pd.DataFrame({"drug": list(dict.fromkeys(selected_compounds))})
    selected = requested.merge(table, on="drug", how="left", validate="one_to_one")
    selected["prediction_status"] = np.where(selected["rank"].notna(), "result_available", "no_prediction")
    selected.sort_values("rank", na_position="last").to_csv(out / "selected_compound_rankings.csv", index=False)
    return selected
```

表征只含部分基因且已有明确补零方案时，执行者在副本中将缺失检查改成显式记录，再使用官方补零路径；报告覆盖率以每个 fold 的全部必需基因为分母。签名方向和药物 ID 映射在调用前检查。上述函数的输入签名需是已映射、按官方清洗规则规范的 HGNC symbol；不要依赖运行时清洗解决重复别名。

## 4. 结果处理

- 保存全部原生排名及本次药物库的覆盖表。模型类未覆盖的化合物标为 `no_prediction`；在覆盖范围内按原生 rank 排序，保留同分/并列关系。库内排序与全模型 rank 分列。
- `logit` 是模型分数，`prob` 是分类 softmax 值，`rank=1` 表示优先级最高。输出原生分数/排名图和输入签名图，配套数值表、manifest、版本和失败记录。
- 本路线的 `result_type=ranking`；恢复方向分类标为“原生候选排名，未计算逐药转录恢复分数”。`Top 30 affected genes` 标为 `not_applicable`，原因是模型没有逐药基因响应；输入签名 Top 基因单列为“状态转换特征”。
- 用户同时要求逐药恢复分类和 Top 30 时，在模型选择阶段说明上述输出差异，列出表达预测后端供其选择。选用两条路线时分别运行，以药物 ID 对照结果。
