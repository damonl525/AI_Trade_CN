# AI_Trade_CN — AI 驱动的 A 股量化交易平台

轻量级、纯本地、零门槛的 A 股 ETF 量化交易系统。数据抓取 → 信号生成 → 回测验证 → 模拟盘追踪 → 实盘建议，一条龙覆盖。

## 设计哲学

- **熊市不亏，牛市跟上** — 动态择时引擎，不是「买或不买」的二元开关，而是 0-100% 连续油门
- **不依赖黑盒 AI 做决策** — LLM 只负责把数据翻译成人话（AI 日报），不参与买卖判断
- **你先用模拟盘跑 1-2 个月，再决定真钱跟不跟**

## 快速上手

```bash
# 统一入口: bash run.sh （自动清除 PYTHONPATH 避免环境冲突）
# 等价于:  PYTHONPATH= uv run python main.py <args>

# 1. 安装依赖
uv sync

# 2. 抓数据（首次全量，后续增量）
bash run.sh fetch

# 3. 看今天能进场吗
bash run.sh entry

# 4. 创建你的第一个模拟盘账户
bash run.sh account create 模拟盘1 --cash 100000

# 5. 根据动态策略调仓（先 dry-run）
bash run.sh trade 模拟盘1 --dry
```

## 每日操作流

```bash
# ① 收盘后：刷新行情（拉今日收盘价，后面所有盈亏基于这个）
bash run.sh live

# ② 看账户盈亏（已自动计算：总资产−总成本）
bash run.sh account status 模拟盘1

# ③ 明天怎么调？（先 dry-run 看建议）
bash run.sh trade auto --dry

# ④ 确认后执行
bash run.sh trade auto 模拟盘1
```

> **先 `live` 再 `status`** — 不跑 `live` 的话 `status` 拿的还是昨天收盘价，盈亏是假的。

## 核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 📡 **数据** | `data/fetcher.py` | AKShare 抓取 + Parquet 本地存储 |
| 🧠 **信号** | `engine/signals.py` | 动量轮动 / MA 交叉 / 布林线 / 均值回归 |
| ⏱️ **择时** | `engine/timing.py` | 趋势+动量+波动率 三因子动态仓位 (0-100%) |
| 🛡️ **风控** | `engine/risk.py` | 回撤熔断 / 单只止损 −8% / 止盈提醒 +20% |
| 🔬 **优化** | `engine/optimize.py` | 网格搜索 + 前向验证 |
| 🤖 **AI 日报** | `ai_summary.py` | DeepSeek / 智谱 / OpenAI 解读行情 |
| 📈 **模拟盘** | `engine/paper_trader.py` | 多账户 SQLite 持久化，动态策略驱动 |
| 📊 **回测** | `engine/backtest.py` | A 股真实成本 (万5佣金+千0.5印花税) |

## 完整命令

### 数据
```bash
uv run python main.py fetch    # 拉取行情（首次全量）
uv run python main.py sector   # 板块热度
```

### 信号
```bash
bash run.sh entry           # 零持仓入场检查 (0-10分)
bash run.sh pos             # 当前动态仓位
bash run.sh signal          # 今日持仓建议 + 仓位
bash run.sh signal --tk 3   # 持仓 3 只
bash run.sh live            # 一键刷新数据+出信号
```

### 模拟盘
```bash
bash run.sh account create <名称> --cash 100000  # 创建模拟账户
bash run.sh account create <名称> --cash 50000 --type live  # 创建实盘账户
bash run.sh account create <名称> --cash 30000 --cost 50000 --type live  # 实盘(成本≠现金)
bash run.sh account list     # 所有账户
bash run.sh account status [名称]  # 账户详情(含成本/净盈亏)
bash run.sh account reset <名称>   # 重置账户
bash run.sh account update-cost <名称> <新成本>  # 更新实盘总成本
bash run.sh account delete <名称>     # 删除账户(含所有记录)
```

### 调仓 (trade)

#### 策略驱动 (auto)
```bash
bash run.sh trade auto              # 全部模拟盘依次调仓
bash run.sh trade auto 模拟盘1       # 只调指定账户
bash run.sh trade auto --dry        # 只看建议，不执行
bash run.sh trade auto 模拟盘1 --dry # 指定账户只看不执行
```

#### 手动录入 (manual) — 实盘操作后追记
```bash
# 格式: trade manual <账户> <buy|sell> <代码> <价格> <股数>
bash run.sh trade manual 真实1 buy 510050 2.718 1000
bash run.sh trade manual 真实1 sell 588000 4.850 500
```

> **手动录入会记录到交易历史，自动更新持仓和现金，并保存 NAV 快照。**
> 手续费自动计算 (万 5 佣金 + 千 0.5 卖印花税 + 最低 5 元)。

### 查看
```bash
bash run.sh history [账户]   # 交易历史
bash run.sh nav [账户]       # 净值曲线
bash run.sh advice           # 所有账户动态策略建议
```

### 回测
```bash
bash run.sh bt               # 标准回测
bash run.sh btlist           # 所有策略回测
bash run.sh dynbt            # 动态 vs 二元 vs 无择时
bash run.sh dynbt --all      # 全动量公式对比
bash run.sh optimize         # 参数优化
```

### AI 摘要
```bash
bash run.sh daily            # AI 日报 (DeepSeek)
bash run.sh daily --brief    # 纯数据无 AI
```

### ETF 池管理
```bash
bash run.sh pool list          # 查看当前 ETF 池
bash run.sh pool add <代码> <名称>  # 添加 ETF
bash run.sh pool remove <代码>      # 删除 ETF
bash run.sh pool reset              # 重置为默认池
```

> 默认池含 16 只 ETF：上证50 / 沪深300 / 中证500 / 创业板 / 科创50 / 证券 / 医药 / 酒/消费 / 红利 / 国债 / 纳指ETF / 纳指100 / 机器人 / 黄金 / AI智能 / 能源。增删存在 `data/etf_pool.json`，另一台电脑 `git pull` 同步。

### 配置 AI Provider
```powershell
# DeepSeek (默认)
$env:AI_SUMMARY_KEY="sk-xxx"
$env:AI_SUMMARY_MODEL="deepseek-v4-flash"
$env:AI_SUMMARY_BASE_URL="https://api.deepseek.com"
$env:AI_SUMMARY_PROVIDER="deepseek"

# 智谱 (免费)
$env:AI_SUMMARY_KEY="你的智谱Key"
$env:AI_SUMMARY_PROVIDER="zhipu"

# OpenAI
$env:AI_SUMMARY_KEY="sk-xxx"
$env:AI_SUMMARY_PROVIDER="openai"
$env:AI_SUMMARY_BASE_URL="https://api.openai.com/v1"
$env:AI_SUMMARY_MODEL="gpt-4o-mini"
```

## 回测结果

### 动量公式对比（全期 2020-2026，动态择时叠加）

| 方案 | 总收益 | 年化 | 最大回撤 | 夏普 |
|------|:--:|:--:|:--:|:--:|
| 🏆 **双动量 (Dual)** | **+73.5%** | **13.7%** | **−10.4%** | **1.10** |
| 基准 20日 | +70.1% | 13.2% | −10.4% | 1.06 |
| Frog-in-the-Pan | +52.1% | 10.2% | −10.8% | 0.87 |
| 波动率加权 | +38.8% | 7.9% | −11.1% | 0.74 |
| 多窗口融合 | +38.4% | 7.9% | −10.7% | 0.69 |

> 参数: lookback=10, top_k=3, 调仓 40 天。数据窗口: 2020-01-01 ~ 2026-06-26

### 训练/测试分离验证 (OOS)

为防过拟合，用 **2016-2021 训练，2022-2026 验证**：

| | 训练期 (2016-2021) | 测试期 (2022-2026) |
|---|---|---|
| 训练最优 (Frog/10/3) | +166.0% | +55.2% ⚠️ |
| 🏆 双动量/10/3 | +146.6% (排第5) | **+77.9%** ✅ |

> 双动量在训练集只排第 5，但在测试集排第 1——说明策略泛化能力真实存在，不是过拟合产物。  
> 训练冠军 Frog 在测试期腰斩，验证了「不碰跌的 ETF」这条硬规则在熊市中的价值。

### 验证命令

```bash
# 全期回测（五种动量公式对比）
bash run.sh dynbt --all

# 训练/测试分离验证（防过拟合）
PYTHONPATH= uv run python train_test_backtest.py
```

## 交易成本

已内建 A 股实际规则：

- 佣金：万 5 (最低 5 元/笔)
- 印花税：千 0.5 (仅卖出)
- 证券代码自动识别：沪(600xxx / 688xxx / 510xxx) / 深(000xxx / 002xxx / 300xxx / 159xxx)

## 依赖

- Python 3.10+
- [AKShare](https://github.com/akfamily/akshare) — A 股数据
- [bt](https://github.com/pmorissette/bt) — 回测引擎
- pandas / numpy / matplotlib / openai

## License

MIT

## 作者

[Bojian](https://github.com/damonl525)
