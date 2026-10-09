# CellOracle 隔离环境搭建记录

测试 AIVC skill 的 KO/OE 分支需要 CellOracle。CellOracle 的依赖链在 Windows 上
不能直接 `pip install`，这里记录实际可用的搭建路径，便于复现。

## 目标环境

- 路径：`D:/WorkBuddy/env/celloracle310`
- 解释器：Python 3.10.22（`D:/WorkBuddy/env/python-runtimes/cpython-3.10.22-windows-x86_64-none`）
- 包管理：`D:/WorkBuddy/env/Scripts/uv.exe`
- 关键约束：本机**没有 MSVC / gcc / clang**，只有 32G 内存、8G 显存

## 为什么不能直接装

`pip install celloracle` 会拉进两个只有源码包的依赖：

| 包 | 问题 |
| --- | --- |
| `velocyto>=0.17` | 只有 sdist，且唯一的 Cython 模块 `speedboosted` 需要 C 编译器 |
| `gimmemotifs<=0.17.2` | 只有 sdist，依赖链里的 `pysam` / `pybedtools` / `pybigwig` / `htseq` 在本机没有可用轮子 |

conda-forge 与 bioconda 都没有 `velocyto` 的 win-64 构建，所以没有"直接下载预编译包"这条路。

## 实际做法

### 1. 装基础依赖

```bash
cd D:/WorkBuddy/env
./Scripts/uv.exe pip install --python D:/WorkBuddy/env/celloracle310/Scripts/python.exe \
  "numpy==1.26.4" cython setuptools wheel
./Scripts/uv.exe pip install --python D:/WorkBuddy/env/celloracle310/Scripts/python.exe \
  --no-deps celloracle==0.20.0
./Scripts/uv.exe pip install --python D:/WorkBuddy/env/celloracle310/Scripts/python.exe \
  scipy "numba>=0.50.1" "matplotlib<3.7" seaborn scikit-learn "h5py>=3.1.0" \
  "pandas>=1.0.3,<=1.5.3" "pyarrow>=0.17" "tqdm>=4.45" "igraph>=0.10.1" louvain \
  "anndata>=0.7.5,<=0.10.8" "scanpy>=1.6" joblib umap-learn goatools genomepy ipython ipywidgets
```

`--no-deps` 是为了跳过 velocyto / gimmemotifs，其余依赖单独装。

### 2. 用 zig 编译 velocyto 的 Cython 模块

本机没有编译器，但 pip 包 `ziglang` 自带一个完整的 zig 工具链（含 clang），可以当 C 编译器用。

关键点：velocyto sdist 里自带的 `speedboosted.c` 是旧版 Cython 生成的，
在 Python 3.10 上会报 `expression is not assignable` 和 `no member named 'tp_print'`，
**必须用 Cython 3 从 `speedboosted.pyx` 重新生成 C**。

```bash
# zig 在 uv 缓存里，装完 ziglang 后从这里取；也可以直接 pip install ziglang 后取
ZIG="<...>/ziglang/zig.exe"
SP="D:/WorkBuddy/env/celloracle310/Lib/site-packages"
PYROOT="D:/WorkBuddy/env/python-runtimes/cpython-3.10.22-windows-x86_64-none"

mkdir -p D:/WorkBuddy/env/velocyto_build && cd D:/WorkBuddy/env/velocyto_build
cp <velocyto sdist>/velocyto/speedboosted.pyx .
D:/WorkBuddy/env/celloracle310/Scripts/python.exe -m cython -3 speedboosted.pyx -o speedboosted_new.c
"$ZIG" cc -shared -O2 -DNDEBUG -DMS_WIN64 -D_CRT_SECURE_NO_WARNINGS \
  -I"$SP/numpy/core/include" -I"$PYROOT/include" \
  -o speedboosted.pyd speedboosted_new.c \
  -L"$PYROOT/libs" -lpython310
```

校验：把 `_colDeltaCor` 的结果和 numpy 手写的 Pearson 参照实现对比，
非对角元素最大绝对误差 `1.7e-16`（浮点舍入量级），`_colDeltaCorpartial` 与全量版本一致。

然后把 velocyto 的 Python 源码连同 `speedboosted.pyd` 放进 site-packages：

```bash
cp -r <velocyto sdist>/velocyto "$SP/velocyto"
rm -f "$SP/velocyto/speedboosted.c" "$SP/velocyto/speedboosted.pyx"
cp D:/WorkBuddy/env/velocyto_build/speedboosted.pyd "$SP/velocyto/"
```

velocyto 原始 `__init__.py` 会连带导入计数模块（需要 `pysam`），已替换为只导入
CellOracle 实际使用子模块的版本，原文件保留在同目录 `__init__.py.orig`。
`estimation.py` / `diffusion.py` / `serialization.py` 均为 velocyto 官方源码，未改动。

### 3. gimmemotifs / pybedtools 占位

CellOracle 的 `celloracle.motif_analysis` 在 import 阶段需要 `gimmemotifs` 和
`pybedtools`，但这条路径是"从基因组序列 + motif 库扫描构建 base GRN"。
本流程直接加载 CellOracle 官方预构建的人类 promoter base GRN，**不走这条路径**。

因此 `site-packages/gimmemotifs/` 与 `site-packages/pybedtools/` 是占位模块：
符号存在、import 可以通过，一旦真正被调用会抛 `NotImplementedError`。
占位模块不参与任何计算，也不会静默返回结果。

## 已知非阻断告警

导入 CellOracle 时会打印：

- `which: no R in ...` — `test_R_libraries_installation()` 找不到 R，本流程不用 R
- `1 package does not meet CellOracle requirement. Your jupyter version is not_found` — 仅提示，不影响计算路径

## 结论

- CellOracle 0.20.0 的**计算路径**（GRN 推断、信号传播模拟、embedding 位移）
  用的是官方源码和官方预构建 base GRN，没有替换。
- 被替换的只有三处，且都在这条路径之外：
  1. velocyto 的编译产物（用官方 pyx 重新生成 C 后用 zig 编译）；
  2. velocyto 的 `__init__` 导入范围（跳过需要 pysam 的计数模块）；
  3. gimmemotifs / pybedtools 两个未使用依赖的占位模块。
