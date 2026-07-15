# PROJECT_RULES.md — CSI Consumer Index Quantitative Analysis Toolkit

> **Single Source of Truth.** 所有开发规范、架构决策、代码约定以此文件为准。
> 新对话开始时，先读此文件恢复上下文，再进行任何开发操作。
> **规则只追加，不覆盖历史原则。** 变更时在末尾 `## 变更日志` 记录。

---

## 一、项目目标

### 定位
研究级量化金融分析工具，面向 **CSI 消费指数 (sz399932)** 的端到端风险建模与策略回测。

### 核心功能
1. **数据管道** — 容错的多源数据获取（AkShare 双端点 + 重试 + 本地缓存 + 估值代理链）
2. **波动率建模** — GARCH 族模型（网格搜索 + BIC 选模 + 非对称模型 + 滚动窗参数漂移监控）
3. **风险模拟** — 10,000 路径蒙特卡洛（GBM / GARCH 时变 / 马尔可夫区制转换 / 历史自助 / OU 择时）
4. **风险指标** — VaR/CVaR 计算 + Kupiec 统计检验 + 终值分布分析

### 技术栈
- **语言:** Python 3.9+
- **核心库:** `numpy`, `pandas`, `arch` (GARCH), `akshare` (数据), `scipy` (统计), `matplotlib` (可视化)
- **辅助库:** `scikit-learn`, `xgboost` (ML 方向预测，非主线)
- **环境:** macOS / Linux, conda / venv

### 资产无关性
架构设计为跨资产可迁移：将 CSI 消费指数替换为外汇对或利率标的，GBM 内核、模拟引擎、VaR 层、数据管道架构均保持不变。利率类标的需要将 GBM 扩散替换为 Hull-White 或 CIR 单因子模型。

---

## 二、开发原则

### 1. 正确性优先 (Correctness Over Speed)
- 任何变更必须先通过现有测试
- 模型参数的变更必须重新跑完整网格搜索验证
- GARCH 收敛失败时不允许静默降级——必须显式标注或抛出异常

### 2. 可维护性优先 (Maintainability)
- 共享工具统一放在 `modules/core.py`，禁止在各脚本间复制粘贴代码
- 每个脚本一个明确职责，脚本之间通过 `modules/core.py` 解耦
- 类型标注 (`from __future__ import annotations`) 为强制要求

### 3. 小步迭代 (Small Iterations)
- 每次变更仅改一个模块或一个函数的单一职责
- 禁止"改完所有文件一次性提交"——每个逻辑独立的改动单独 commit

### 4. 单模块开发 (Single Module Focus)
- 一次只开发或修改一个模块
- 模块完成后才能进入下一模块
- 严禁同时修改两个及以上独立模块

### 5. 禁止 Vibe Coding (No Speculation-Driven Development)
- 所有新增功能必须先有明确需求或问题描述
- 禁止"顺手加一个功能"式的自由发挥
- 模型方法论选择必须有统计准则支撑（如 BIC、Kupiec 检验），不能凭直觉

### 6. 禁止一键生成整个项目 (No Batch Generation)
- 严禁单次对话生成超过一个模块的代码
- 严禁使用"把全部文件改完再测试"的工作方式

### 7. 禁止跳步开发 (No Step Skipping)
- 严格遵循 分析 → 设计 → 实现 → 测试 → 文档更新 → Git 提交 → 等待确认 流程
- 每一阶段完成并验证后，才能进入下一阶段

---

## 三、开发流程

每个开发周期严格执行以下七步：

```
分析 (Analysis) → 设计 (Design) → 实现 (Implementation) → 测试 (Testing) → 文档更新 (Docs) → Git提交 (Commit) → 等待确认 (Await Confirmation)
```

### 3.1 分析 (Analysis)
- 阅读相关模块的现有代码
- 确认问题范围：是新增功能、修复 bug、还是重构
- 如有 ADR，阅读相关 `docs/adr/` 文件
- 输出：对问题的理解描述 + 影响范围评估

### 3.2 设计 (Design)
- 确定修改方案（文件、函数、参数变更）
- 评估对现有模型输出的影响（如：改 GARCH 分布是否影响 Monte Carlo 结果）
- 如需架构级决策，写入 ADR
- 输出：修改计划

### 3.3 实现 (Implementation)
- 按设计实现，不超范围
- 遵守本文件中的代码规范

### 3.4 测试 (Testing)
- 每个修改的脚本必须语法编译通过
- 模型变更必须跑完整端到端验证
- 输出关键指标确认无回归（如 VaR 值是否合理、GARCH 是否收敛）
- 测试命令参考：
  ```bash
  # 语法检查
  python -c "import py_compile; py_compile.compile('path/to/file.py', doraise=True)"

  # GARCH 相关测试（需设 ARCH_DISABLE_NUMBA=1 如环境有 numba 冲突）
  ARCH_DISABLE_NUMBA=1 python garch_gridsearch_sz399932.py

  # Monte Carlo 端到端
  ARCH_DISABLE_NUMBA=1 MPLBACKEND=Agg python monte_carlo_csi_consumer.py
  ```

### 3.5 文档更新 (Docs)
- 更新 README.md（如影响使用说明或输出）
- 更新 `消费项目_面试_代码与JD话术.md`（如影响面试话术数据）
- 如果成果显著，更新 `DEVELOPMENT_LOG.md`

### 3.6 Git 提交 (Commit)
- 仅提交相关文件，不批量提交无关变更
- Commit message 使用英文，格式：`type: summary`（feat/fix/chore/docs/refactor）
- 面试文档 (`消费项目_面试_代码与JD话术.md`) **不入库** — 留在本地 `.gitignore` 范围

### 3.7 等待确认 (Await Confirmation)
- 提交后暂停，等待用户 review
- 不自动进入下一任务或下一个模块

---

## 四、模块开发规则

### 4.1 一次一模块
- 每个开发周期只针对一个模块
- 模块单位 = 一个 `.py` 脚本或 `modules/` 下的一个子模块

### 4.2 当前模块清单

| 模块 | 路径 | 职责 |
|------|------|------|
| core | `modules/core.py` | 共享工具：数据获取、GARCH 拟合、波动率辅助函数 |
| rolling_garch | `modules/rolling_garch.py` | 滚动窗 GARCH 参数漂移分析 |
| garch_baseline | `garch_sz399932.py` | GARCH(1,1) 基线拟合 + 波动率聚集可视化 |
| garch_gridsearch | `garch_gridsearch_sz399932.py` | 网格搜索 + 非对称模型对比 |
| monte_carlo | `monte_carlo_csi_consumer.py` | 蒙特卡洛模拟 + VaR/CVaR + Kupiec |
| ml_direction | `ml_direction_csi_consumer.py` | ML 方向预测（非主线，不主动提及） |
| comparison | `comparison_csi_vs_hs300.py` | CSI 消费 vs 沪深 300 基准对比 |

### 4.3 模块完成标准
1. ✅ 功能无报错
2. ✅ 相关测试通过（语法 + 端到端）
3. ✅ README.md 更新（如输出改变）
4. ✅ 面试话术文档更新（如指标改变）
5. ✅ Commit message 描述清晰

---

## 五、架构规范

### 5.1 分层架构

```
┌──────────────────────────────────────────────────────────┐
│  ③ 模拟/风险层                                           │
│  monte_carlo_csi_consumer.py                             │
│  ┌────────────────────────────────────────────────────┐  │
│  │ GBM → GARCH time-varying → Markov regime-switching │  │
│  │ → Historical bootstrap → OU strategic DCA          │  │
│  │ Output: terminal P&L, VaR/CVaR, Kupiec p-value     │  │
│  └────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────┤
│  ② 波动率层                                              │
│  garch_sz399932.py / garch_gridsearch_sz399932.py        │
│  modules/rolling_garch.py                                │
│  ┌────────────────────────────────────────────────────┐  │
│  │ GARCH(1,1) → EGARCH/GJR-GARCH → rolling α/β/ω     │  │
│  │ Output: cond_vol, forecast_vol, leverage params    │  │
│  └────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────┤
│  ① 数据层                                                │
│  modules/core.py                                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │ AkShare dual-endpoint → retry + backoff → CSV cache│  │
│  │ PE proxy chain (tier 0-5) with source_tier tag     │  │
│  │ Output: clean daily price + valuation proxy        │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 5.2 依赖方向
- `modules/core.py` ← **所有脚本的唯一共享依赖**
- 上层脚本可以 import 下层模块，反之不行
- Monte Carlo 脚本本身是顶层入口，不向其他脚本暴露 import 接口（工具函数如 `kupiec_test` 除外）

### 5.3 数据流
```
AkShare API → core.py (fetch + clean + cache) → GARCH scripts → MC scripts → PNG + Console Output
```

### 5.4 配置集中化
- 全局常量（`CSI_CONSUMER_SYMBOL`, `TRADING_DAYS_PER_YEAR`）在 `modules/core.py` 定义
- 模拟参数（`SIM_PATHS`, `MONTHLY_INVESTMENT`）在 `monte_carlo_csi_consumer.py` 顶部定义
- 禁止在函数体内埋魔法数字

---

## 六、代码规范

### 6.1 Python 版本与语法
- Python 3.9+
- 文件头：`#!/usr/bin/env python3`
- 必须包含：`from __future__ import annotations`
- 类型标注：所有函数签名必须有类型标注

### 6.2 导入顺序
```python
# 1. 标准库
from __future__ import annotations
from pathlib import Path
import time

# 2. 第三方库
import numpy as np
import pandas as pd

# 3. 本地模块
from modules.core import CSI_CONSUMER_SYMBOL, fetch_index_daily
```

### 6.3 命名规范
| 类型 | 规范 | 示例 |
|------|------|------|
| 全局常量 | `UPPER_SNAKE` | `TRADING_DAYS_PER_YEAR`, `RNG_SEED` |
| 函数名 | `snake_case` | `compute_var_cvar`, `fetch_index_daily` |
| 变量名 | `snake_case` | `annual_mean_return`, `monthly_vol_array` |
| 数据类 | `PascalCase` | `SimulationStats` |
| 模块文件名 | `snake_case` | `rolling_garch.py` |
| 私有函数 | `_prefix` | `_clean`, `_try_fetch` |

### 6.4 量化专项命名约定
- `annual_*` = 年化值
- `daily_*` = 日频值
- `monthly_*` = 月频值
- `log_return` / `returns_pct` = 对数收益 / 百分比收益（×100）
- `cond_vol` = 条件波动率
- `fc_vol` / `forecast_vol` = 前向预测波动率
- `var_95` / `cvar_95` = VaR/CVaR 在指定置信水平
- `GARCH_CALIBRATION_WINDOW` = GARCH 拟合窗口大小（trading days）

### 6.5 禁止事项
- ❌ 函数体内硬编码路径、参数、阈值
- ❌ `except: pass` — 静默吞异常
- ❌ `import *`
- ❌ 行内注释使用中文（docstring 和模块说明可用中文，但代码注释用英文）
- ❌ 超长行（>120 字符）

---

## 七、测试规范

### 7.1 测试层级

| 层级 | 内容 | 何时执行 |
|------|------|---------|
| **L1 语法** | `py_compile.compile(f, doraise=True)` | 每次代码改动后 |
| **L2 导入** | import 目标模块，确保无缺失依赖 | L1 通过后 |
| **L3 函数验证** | 关键函数的独立调用验证 | 函数新增/修改后 |
| **L4 端到端** | 完整脚本运行，检查输出指标 | 模块完成阶段 |

### 7.2 量化专项测试要求
- **GARCH 变更：** 必须验证 `收敛性`（converged = True）、`持续性`（α+β < 1）、`网格排名`（BIC 排序不变或可解释变化）
- **Monte Carlo 变更：** 必须验证 `VaR 合理性`（不出现负值 VaR）、`Kupiec p-value > 0.05`、`盈利概率横截面合理`（各模式间差异可解释）
- **数据管道变更：** 必须验证 `缓存回退` 有效、`超时保护` 触发正常

### 7.3 已知环境问题
- **ARCH_DISABLE_NUMBA=1:** macOS 上 `llvmlite` 可能与 `numba` 冲突，运行 GARCH 拟合脚本前需设置此环境变量
- **MPLBACKEND=Agg:** 非交互式环境（CI/后台运行）需设置此变量以避免 matplotlib GUI 阻塞

---

## 八、文档规范

### 8.1 必须维护的文档

| 文件 | 状态 | 用途 |
|------|:--:|------|
| `README.md` | ✅ 已有 | 项目概览、快速开始、方法论说明、数据质量声明 |
| `PROJECT_RULES.md` | ✅ 本文件 | 开发规范 SSOT |
| `消费项目_面试_代码与JD话术.md` | ✅ 已有（不入库）| 面试话术稿，严格对齐代码事实 |
| `ARCHITECTURE.md` | ❌ 待建 | 架构详细说明 |
| `DEVELOPMENT_LOG.md` | ❌ 待建 | 开发日志 |
| `docs/adr/` | ❌ 待建 | 架构决策记录 |

### 8.2 文档更新规则
- 代码变更影响输出指标 → 必须更新 README 和面试话术文档
- 方法论改动（如换用 t 分布）→ 必须更新 README "Methodology Choices" 章节
- 数据质量策略变化 → 必须更新 README "Data Quality Notes" 章节

---

## 九、Git 规范

### 9.1 Commit Message 格式
```
<type>: <summary>
```
type: `feat` | `fix` | `chore` | `docs` | `refactor` | `test`

示例：
```
fix: regime-switching uses full-sample calibration, GARCH vol stays rolling-window
chore: regenerate all PNGs with t-distribution GARCH + updated data
feat: add Kupiec POF VaR backtest with p-value output
```

### 9.2 入库排除
- `消费项目_面试_代码与JD话术.md` 不入库（`.gitignore` 或本地留 untracked）
- `__pycache__/` 不入库（已在 `.gitignore`）
- 包含 secret/key 的文件严禁入库

### 9.3 提交粒度
- 一个逻辑改动一个 commit
- 禁止 "fix stuff" / "update" 等无信息量 message
- Commit body 可选，但 fix 类必须解释原因

---

## 十、ADR 规范（待建）

### 10.1 ADR 目录
- `docs/adr/` — 架构决策记录，每个文件记录一次重要技术决策
- 命名格式：`NNNN-title-with-dashes.md`（如 `0001-use-bic-over-aic-for-model-selection.md`）

### 10.2 ADR 模板
```markdown
# ADR-NNNN: Title

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Deprecated | Superseded

## Context
## Decision
## Consequences
```

### 10.3 需 ADR 的场景
- 更换核心模型假设（如 Normal → t 分布）
- 改变数据管道架构（如新增 fallback 层）
- 变更标定窗口策略（如全样本 → 滚动窗）
- 引入新的风险度量或验证方法

---

## 十一、重构规范

### 11.1 何时重构
- 同一逻辑在 ≥ 3 处出现
- 函数体 > 50 行
- 参数列表 > 5 个且无数据类封装

### 11.2 重构流程
1. 先跑全量端到端测试，记录基准指标
2. 实施重构
3. 跑全量端到端测试，确认指标无回归
4. 更新受影响的文档
5. 单独的 refactor commit

### 11.3 禁止行为
- ❌ 在重构时顺便加新功能
- ❌ 重构未测试通过就提交
- ❌ 不保留旧基准数据就覆盖输出

---

## 十二、量化专项规范

### 12.1 数据完整性 (Data Integrity)
- **Point-in-Time 原则：** 任何特征工程必须使用 `shift(1)` 滞后，确保不会使用未来信息
- **Look-ahead Bias 检测：** 关键特征需有数值断言检查（如 `check_no_lookahead_bias` 函数）
- **PE/Valuation 代理原则：** 所有估值数据必须带 `source_tier` 质量标记。tier ≥ 2 的代理数据不能声称是真实 PE
- **数据源超时保护：** 第三方 API 调用必须带超时机制（ThreadPoolExecutor + timeout），超时后自动进入下一回退层

### 12.2 模型验证 (Model Validation)
- **GARCH 网格搜索：** 所有 p,q 组合必须检查收敛性；收敛失败的不参与 BIC 排名
- **BIC vs AIC：** 模型排序用 BIC（对参数个数惩罚更重，更适合大样本 + 风控场景）
- **非对称模型：** EGARCH/GJR-GARCH 必须在 t 分布下拟合；如不收敛需显式标注 N/A
- **滚动窗监控：** 滚动窗 GARCH 的持续性（α+β）最大允许值为 0.999；持续逼近 1 应触发重标定警告
- **区制转换标定策略：** 转移矩阵和区制收益率用全样本估计（需完整市场周期）；前向波动率预测用最近 4 年（1008 trading days）滚动窗

### 12.3 风险计算规范 (Risk Calculation Standards)
- **VaR 口径：** 本项目 VaR 为 **5 年持有期终值口径**（非监管口径的 1 日/10 日 VaR），面试和文档中必须明确标注
- **VaR 计算：** `losses = total_investment - final_values`，`VaR = np.percentile(losses, level*100)`
- **CVaR 计算：** `CVaR = losses[losses >= VaR].mean()`（Expected Shortfall）
- **Kupiec 检验：** VaR 模型变更后必须跑 Kupiec POF 检验，p-value < 0.05 表明模型存在系统性偏误
- **风险指标对比：** Monte Carlo 各模式的 VaR/CVaR 必须交叉对比并解释差异来源

### 12.4 可复现性 (Reproducibility)
- 所有随机过程使用固定 seed (`RNG_SEED = 42`)，确保结果可复现
- 模拟路径数 (`SIM_PATHS = 10_000`) 为固定常量，非特殊情况不调整
- 数据缓存文件 (`sz399932_akshare_cache.csv`) 随代码入库，确保离线也可复现

### 12.5 GARCH 标定窗口策略 (Calibration Window Policy)
- 前向波动率预测：使用 `GARCH_CALIBRATION_WINDOW = 1008`（约 4 年，最近交易日）
- 转移矩阵估计（区制转换）：使用 `calibration_window=None`（全样本，需完整牛熊周期）
- 滚动窗参数分析：4 年窗 + 季度步长，50+ 次重估
- 独立基线 GARCH 脚本（`garch_sz399932.py`）：可按需使用全样本或指定窗口

---

## 十三、常见开发陷阱 (Footguns)

这些是历史上踩过的坑，新开发时务必注意：

1. **EGARCH + t 分布 = 大概率不收敛。** 如果需要非对称模型，优先用 GJR-GARCH。
2. **AkShare `stock_zh_index_value_csindex` 会挂起。** 必须用 `ThreadPoolExecutor(timeout=15)` 包裹，否则整个管道卡死。
3. **`annual_volatility` vs `annual_vol_array` 混淆。** 前者是标量（恒定波动率），后者是数组（时变波动率）。调用 `run_monte_carlo` 时两者互斥。
4. **PE 代理链的 `pe_ttm` 值在 0.5-2.0 区间（不是真实 PE 的 10-60）。** 任何用到该值的逻辑必须明确其仅具有方向含义，绝对量级无意义。
5. **Monte Carlo 区制转换模型只用 4 年窗标定会导致崩溃式结果。** 转移矩阵必须从全样本条件波动率估计。

---

## 变更日志

| 日期 | 变更内容 | 类型 |
|------|---------|------|
| 2026-07-15 | 初始创建 PROJECT_RULES.md | chore |
| — | （后续变更在此追加，不修改以上内容） | — |

---

*本文件为项目唯一权威开发规范。任何开发行为与此文件冲突时，以此文件为准。*
