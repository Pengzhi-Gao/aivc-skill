# AIVC 执行手册

## 四阶段

1. 选择 KO/OE/药筛及结果类型。
2. 检查输入和模型契约。
3. 运行每个目标/药物条件并落盘原始结果。
4. 生成表格、图、质量记录和 report.md。

## 运行环境

各后端跑在各自隔离的解释器里，按 `backend-environments.md` 选环境，不要在主环境里跑
CellOracle。启动前先确认解释器路径和后端版本，并写进 `manifest`。

## 运行记录

每次运行保存：

```text
results/<run_id>/
  manifest.yaml
  provenance.json
  raw/
  tables/
  figures/
  validation.json
  errors.jsonl
  report.md
```

批量药筛以 `(drug, dose, time, cell_type)` 为最小运行单元。已完成条件写入 checkpoint，失败条件写入 errors，不用空结果覆盖。

## 报告内容

`report.md` 只包含：

- 任务和后端
- 输入文件、预处理和参数
- 完成/跳过/失败条件
- 结果文件路径
- 输出类型和单位
- 覆盖率、OOD、重复性或其他技术检查
- 未独立验证项目
- 未评估项目和原因

不自动撰写疾病机制、疗效、因果、靶点或实验优先级。

## 失败状态

- `no_prediction`: 后端不覆盖目标
- `failed`: 运行报错
- `not_applicable`: 数据或任务不适用
- `partial`: 部分条件完成
- `unvalidated`: 结果已生成但无独立验证

保留可运行条件的原始结果。只有输入无法解释、后端无输出、结果完全退化或关键字段缺失时阻断对应条件。