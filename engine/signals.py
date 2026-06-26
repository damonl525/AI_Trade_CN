"""策略信号模板库

每个策略是一个 backtrader.Strategy 子类。
命名规范: S_<描述> (S_ = Signal)
"""

import backtrader as bt


# ════════════════════════════════════════════
# 1. 双均线交叉 (经典基准)
# ════════════════════════════════════════════
class S_MA_Cross(bt.Strategy):
    """短均线上穿长均线 → 买入, 下穿 → 卖出

    参数:
        fast (5):  短均线周期
        slow (20): 长均线周期
    """
    params = (("fast", 5), ("slow", 20),)

    def __init__(self):
        ma_fast = bt.ind.SMA(self.data.close, period=self.p.fast)
        ma_slow = bt.ind.SMA(self.data.close, period=self.p.slow)
        self.crossover = bt.ind.CrossOver(ma_fast, ma_slow)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()


# ════════════════════════════════════════════
# 2. 单均线 + 缓冲 (减少假突破)
# ════════════════════════════════════════════
class S_MA_Buffer(bt.Strategy):
    """价格突破均线一定比例才交易，减少震荡磨损

    参数:
        period (20): 均线周期
        buffer (0.02): 缓冲 2%
    """
    params = (("period", 20), ("buffer", 0.02),)

    def __init__(self):
        self.ma = bt.ind.SMA(self.data.close, period=self.p.period)

    def next(self):
        price = self.data.close[0]
        ma = self.ma[0]
        if not self.position:
            if price > ma * (1 + self.p.buffer):
                self.buy()
        elif price < ma * (1 - self.p.buffer):
            self.close()


# ════════════════════════════════════════════
# 3. 动量 + 均线过滤器
# ════════════════════════════════════════════
class S_Momentum_MA(bt.Strategy):
    """N日动量 > 阈值 AND 价格 > 均线 → 买入

    参数:
        mom_period (20): 动量回看天数
        mom_thresh (0.03): 动量阈值 3%
        ma_period (60): 均线过滤
    """
    params = (("mom_period", 20), ("mom_thresh", 0.03), ("ma_period", 60),)

    def __init__(self):
        self.mom = bt.ind.ROC(self.data.close, period=self.p.mom_period) / 100
        self.ma = bt.ind.SMA(self.data.close, period=self.p.ma_period)

    def next(self):
        if not self.position:
            if self.mom[0] > self.p.mom_thresh and self.data.close[0] > self.ma[0]:
                self.buy()
        elif self.mom[0] < 0:
            self.close()


# ════════════════════════════════════════════
# 4. 布林线回归 (下轨买 / 中轨卖)
# ════════════════════════════════════════════
class S_Bollinger(bt.Strategy):
    """触下轨反弹买入, 回中轨卖出

    参数:
        period (20): 布林带周期
        dev (2.0): 标准差倍数
    """
    params = (("period", 20), ("dev", 2.0),)

    def __init__(self):
        self.boll = bt.ind.BollingerBands(self.data.close, period=self.p.period, devfactor=self.p.dev)
        self.mid = self.boll.mid
        self.bot = self.boll.bot

    def next(self):
        if not self.position:
            if self.data.close[0] <= self.bot[0] and self.data.close[-1] > self.bot[-1]:
                self.buy()
        elif self.data.close[0] >= self.mid[0]:
            self.close()


# ════════════════════════════════════════════
# 5. 突破 + ATR 止损 (趋势跟随)
# ════════════════════════════════════════════
class S_Breakout_ATR(bt.Strategy):
    """N日高点突破买入, ATR动态止损

    参数:
        lookback (20): 突破回看天数
        atr_period (14): ATR周期
        atr_mult (2.0): 止损 = 买入价 - N*ATR
    """
    params = (("lookback", 20), ("atr_period", 14), ("atr_mult", 2.0),)

    def __init__(self):
        self.highest = bt.ind.Highest(self.data.high, period=self.p.lookback)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        self.stop_price = None

    def next(self):
        if not self.position:
            if self.data.close[0] >= self.highest[-1]:
                self.buy()
                self.stop_price = self.data.close[0] - self.p.atr_mult * self.atr[0]

        if self.position and self.stop_price is not None:
            if self.data.close[0] < self.stop_price:
                self.close()
                self.stop_price = None
            else:
                new_stop = self.data.close[0] - self.p.atr_mult * self.atr[0]
                self.stop_price = max(self.stop_price, new_stop)  # 止盈上移


# ════════════════════════════════════════════
# 6. ETF轮动底座 (多标的)
# ════════════════════════════════════════════
class S_Rotation(bt.Strategy):
    """等权持有N只ETF, 按动量(M)排名, 持有Top K只, 每M天调仓

    参数:
        symbols: list 标的代码(不用于回测,仅用于meta)
        top_k (3): 持仓数
        lookback (20): 动量回看窗口
        rebal_freq (20): 调仓频率(交易日)
    """
    params = (("top_k", 3), ("lookback", 20), ("rebal_freq", 20),)

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f"  [{dt}] {txt}")

    def __init__(self):
        self.day = 0
        self.roc = {}
        for i, d in enumerate(self.datas):
            self.roc[d._name] = bt.ind.ROC(d.close, period=self.p.lookback)

    def next(self):
        self.day += 1
        if self.day % self.p.rebal_freq != 0:
            return

        # 计算动量排名
        scores = {}
        for i, d in enumerate(self.datas):
            if len(d) > self.p.lookback:
                scores[d._name] = (self.roc[d._name][0], i)

        top = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:self.p.top_k]
        top_names = {t[0] for t in top}

        # 清仓不在 Top K 的
        for i, d in enumerate(self.datas):
            pos = self.getposition(d)
            if pos.size > 0 and d._name not in top_names:
                self.close(data=d)

        # 等权买入所有 Top K
        for i, d in enumerate(self.datas):
            if d._name in top_names:
                target = self.broker.getvalue() / self.p.top_k
                price = d.close[0]
                size = target / price
                pos = self.getposition(d)
                current_size = pos.size if pos else 0
                diff = size - current_size
                if diff > 0:
                    self.buy(data=d, size=int(diff // 100 * 100))
                elif diff < 0:
                    self.sell(data=d, size=int(-diff // 100 * 100))
