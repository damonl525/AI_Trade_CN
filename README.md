# AI_Trade_CN — AI 驱动的 A 股量化交易平台

轻量级、纯本地、零门槛的 A 股 ETF 量化交易系统。数据抓取 → 信号生成 → 回测验证 → 模拟盘追踪 → 实盘建议，一条龙覆盖。

## 设计哲学

- **熊市不亏，牛市跟上** — 动态择时引擎，不是「买或不买」的二元开关，而是 0-100% 连续油门
- **不依赖黑盒 AI 做决策** — LLM 只负责把数据翻译成人话（AI 日报），不参与买卖判断
- **你先用模拟盘跑 1-2 个月，再决定真钱跟不跟**

## 快速上手

```bash
# 1. 安装依赖
uv sync

# 2. 抓数据（首次全量，后续增量）
uv run python main.py fetch

# 3. 看今天能进场吗
uv run python main.py entry

# 4. 创建你的第一个模拟盘账户
uv run python main.py account create 模拟盘1 --cash 100000

# 5. 根据动态策略调仓（先 dry-run）
uv run python main.py trade 模拟盘1 --dry
```

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
uv run python main.py entry           # 零持仓入场检查 (0-10分)
uv run python main.py pos             # 当前动态仓位
uv run python main.py signal          # 今日持仓建议 + 仓位
uv run python main.py signal --tk 3   # 持仓 3 只
uv run python main.py live            # 一键刷新数据+出信号
```

### 模拟盘
```bash
uv run python main.py account create <名称> --cash 100000  # 创建账户
uv run python main.py account create <名称> --cash 50000 --type live  # 实盘账户
uv run python main.py account list     # 所有账户
uv run python main.py account status [名称]  # 账户详情
uv run python main.py account reset <名称>   # 重置账户
```

### 调仓 (trade)
```bash
uv run python main.py trade            # 全部模拟盘依次调仓 ⚠️
uv run python main.py trade 模拟盘1    # 只调指定账户 (模拟盘或实盘)
uv run python main.py trade --dry      # 只看全部模拟盘建议，不执行
uv run python main.py trade 模拟盘1 --dry  # 只看指定账户建议，不执行
```

> **注意**：`trade` 不带账户名 = 所有 `type=sim` 的模拟盘都会执行调仓。
> 带实盘账户名 (`trade 真实1`) 也会真实调仓。**实盘操作前务必先 `--dry` 预览！**

### 查看
```bash
uv run python main.py history [账户]   # 交易历史
uv run python main.py nav [账户]       # 净值曲线
uv run python main.py advice           # 所有账户动态策略建议
```

### 回测
```bash
uv run python main.py bt               # 标准回测
uv run python main.py btlist           # 所有策略回测
uv run python main.py mtbt             # 多时间窗回测
uv run python main.py dynbt            # 动态 vs 二元 vs 无择时
uv run python main.py optimize         # 参数优化
```

### AI 摘要
```bash
uv run python main.py daily            # AI 日报 (DeepSeek)
uv run python main.py daily --brief    # 纯数据无 AI
```

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

| 策略 | 总收益 | 年化 | 最大回撤 | 夏普 |
|------|:--:|:--:|:--:|:--:|
| 无择时 | 5.5% | 1.1% | −42.9% | 0.16 |
| 二元择时 (0/50/100) | 51.4% | 9.0% | −18.1% | 0.65 |
| 🏆 **动态择时 (0-100%)** | **60.1%** | **10.3%** | **−11.3%** | **0.88** |

> 2022 熊市：动态择时 −0.9%，几乎不亏。2024 牛市：+20.6%，跟满仓一样。

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
