# AIVC：虚拟干预流程 Skill

用已有的 bulk / 单细胞转录组数据，跑虚拟基因敲除（KO）、虚拟基因过表达（OE）和虚拟药物筛选，把每个条件的结果、参数、质量检查和失败原因完整落盘。

这个 Skill 只负责**把流程跑通并交付可复核的产物**。它不替你判断疾病机制、疗效、因果关系或靶点——科学解释留给使用者。

## 它解决什么问题

做虚拟干预分析，常见的问题不是"没有模型"，而是：

- 手上的数据到底能跑哪条路线，跑不了的部分是什么原因；
- 模型要的输入形态和手头数据对不上（比如把任意表达矩阵喂给只在细胞系上训过的模型）；
- 结果看起来有数，实际不随干预变化（退化输出），却被当成排名交付；
- 每个条件的参数、随机种子、权重版本没有记录，换台机器就复现不出来。

AIVC 把这些问题变成流程里的固定检查项：先路由后端，再检查输入契约，跑完做技术对照，最后按统一目录交付。

## 三条路线与后端

| 任务 | 首选后端 | 备选 | 交付的原生结果 |
| --- | --- | --- | --- |
| 虚拟敲除 KO | CellOracle | 有匹配 KO/CRISPRi 训练数据时用 GEARS | 模拟表达层、`delta_X`、网络变化、状态位移 |
| 虚拟过表达 OE | CellOracle | 有匹配 CRISPRa/OE 训练数据时用 GEARS；Geneformer 只交付 embedding 变化 | 同 KO，OE 值取自所用表达尺度的高分位 |
| 虚拟药筛 | PRnet、chemCPA | UniPert-G2CP、UniCure、DrugReflector | 逐药逐基因预测响应、药物得分排名 |

路由规则里几条硬约束：

- 药筛先分清后端**接不接受用户自己的基线矩阵**。接受外部基线的（PRnet、chemCPA）可以直接用"单细胞矩阵 + 药物库"；不接受外部基线的（UniPert-G2CP）只能用其自带细胞系的对照。
- 药筛结果**必须过退化检查**：药物之间没有区分度时标 `failed`，排名只留作排查，不当结果交付。
- **CellOracle 没有"未扰动向量场"**。`simulate_shift` 传播的是"输入 − 实测"的差值，空扰动时恒为 0。扰动分析要用 `estimate_transition_prob(calculate_randomized=True)` 的 `delta_embedding_random` 作随机化 GRN 对照。
- ORBIT 只做 pathway/program 层面分析，不冒充单基因 KO/OE；BulkFormer 不作为化合物扰动后端；不使用 scTenifoldKnk。

## 安装

Skill 是纯文档 + 流程规则，不含模型代码和权重。把仓库放到 skills 目录即可：

```bash
git clone https://github.com/Pengzhi-Gao/aivc-skill.git ~/.workbuddy-ai/skills/aivc
```

各后端的解释器、权重路径和安装障碍见 [`references/backend-environments.md`](references/backend-environments.md)。不同后端的环境相互隔离，不要混用——例如 CellOracle 锁定 `numpy==1.26.4` / `pandas<=1.5.3`，装不进带新版 scanpy 的主环境。

Windows 上从零搭 CellOracle 的完整步骤（无 C 编译器时用 `ziglang` 自带 zig、用 Cython 3 重新生成 `speedboosted.c`）见 [`docs/environment-celloracle-windows.md`](docs/environment-celloracle-windows.md)。

## 使用

Skill 按 `SKILL.md` 的六步工作流执行：选任务和结果类型 → 检查输入 → 路由后端并运行 → 标准化结果 → 生成下游结果 → 交付产物。

一次运行对应一个结果目录：

```text
results/<run_id>/
  manifest.json        # 任务、后端版本、输入哈希、参数、随机种子
  validation.json      # 覆盖检查、退化检查、对照结果、独立验证状态
  errors.jsonl         # 逐条件失败记录
  raw/                 # 后端原始输出
  tables/              # 标准化表
  figures/             # 图与 figure_manifest.csv
  report.md            # 只写运行事实，不写机制结论
```

## 结果展示

执行时按实际数据生成互补图组，用户也可以指定只画某张图：

- **虚拟敲除**：有方向的基因响应图、响应热图和适用的通路富集；有状态位移时，补充细胞群背景、随机化对照与位移分布；有网络或轨迹数据时再增加网络图和扰动评分。
- **虚拟过表达**：基因响应、热图和适用的富集，加目标表达设定展示；有明确目标状态时增加状态接近程度，有匹配敲除结果时作 KO/OE 并排比较。仅 embedding 后端使用距离分布和候选排名。
- **虚拟药筛**：有效候选排名、评分分布、可用的药物响应热图和退化检查；按数据补充疾病逆转、跨细胞背景评分热图与剂量响应曲线。
- **实测验证**：有匹配真实扰动数据时，展示预测对实测散点图、基因表达对照和方向/误差评价；与纯预测图分开。

每张图保存对应数值表、绘图参数和生成状态。没有合适数据时记录跳过原因；不为单份预测补造 p 值、误差条、存活率或 IC50。详细生成条件、坐标定义、展示检查和文献示例见 [`references/downstream-results.md`](references/downstream-results.md)。

上述内容是后续执行的出图规则；仓库历史示例仍以实际保存的图为准，未补画的扩展图不计为已完成。

## 结果状态

不混淆"没有结果"和"没有独立验证"：

| 状态 | 含义 |
| --- | --- |
| `result_available` | 后端正常输出结果 |
| `unvalidated` | 结果已生成，但缺少独立实验/留出验证 |
| `partial` | 部分条件完成 |
| `no_prediction` | 目标不在后端覆盖范围 |
| `failed` | 运行失败并保留错误 |
| `not_applicable` | 输入或任务不适用 |

## 已验证示例：非小细胞肺癌

`examples/lung_cancer/` 是一轮完整的端到端测试，数据全部来自 GEO，癌种为肺癌：

| 分支 | 数据 | 后端 | 结果 |
| --- | --- | --- | --- |
| 疾病参照 | GSE81089（NSCLC bulk RNA-seq，199 肿瘤 / 19 正常） | Welch t 检验 + BH | 2131 个差异基因，978 landmark 基因覆盖 96.3% |
| 虚拟 KO/OE | GSE198291（LUAD scRNA，QC 后 703 细胞） | CellOracle 0.20.0 | STAT3、JUN 各跑 KO 与 OE，扰动特异位移 0.49–0.70，随机化 GRN 对照 0.28–0.30 |
| 虚拟药筛（LINCS 背景） | A549 子集，458 药 / 2367 条件 | PRnet | 逆转分数 −0.137 ~ +0.227，预测与实测逐条件相关均值 0.705 |
| 虚拟药筛（用户基线） | GSE198291 + 2333 个 FDA 已上市药物 × 3 剂量 | chemCPA / PRnet / UniPert-G2CP | 只有 UniPert-G2CP 未退化（极差 0.307、响应两两相关中位数 0.958） |

这一轮最有价值的结论是负面的：**把用户自己的样本当基线喂给在细胞系上训练过的模型，很容易得到一个看起来有数、实际不随药物变化的排名**。chemCPA 和 PRnet 在这个输入下都退化了，退化检查把它们拦了下来。详细数据见 [`examples/lung_cancer/reports/`](examples/lung_cancer/reports/)，结果图在 [`examples/lung_cancer/figures/`](examples/lung_cancer/figures/)。

## 参考文档

| 文件 | 内容 |
| --- | --- |
| `references/input-contracts.md` | 输入字段、缺口处理、疾病参照的构造要求 |
| `references/model-capabilities.md` | 各后端的实际能力、限制和路由依据 |
| `references/backend-environments.md` | 解释器、权重、先验、已知安装障碍、调用要点 |
| `references/downstream-results.md` | 三类任务的默认图组、条件扩展、统计口径、图形源表与展示验收 |
| `references/execution-playbook.md` | 四阶段执行、结果目录、错误处理和交付格式 |

## 边界

- 所有结果在缺少独立实验或留出数据验证时标 `unvalidated`，不因为"没有验证"而丢弃可运行的结果，也不因为"有结果"而把它写成已验证。
- 不向模型塞不匹配的输入：缺关键字段时列缺口，不用零向量、库均值或假数据补齐。
- 训练/下载公开权重前核对版本、许可证、输入物种和细胞背景。
- 分子对接只在用户选定药物并确认候选蛋白后启动，交付分数和阳性对照，不写"已确认靶点"。

## 许可

MIT，见 [LICENSE](LICENSE)。
