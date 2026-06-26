"""模拟盘引擎 — 多账户 · 持久化 · 动态策略驱动

核心理念:
  - 每个账户独立持仓、独立资金
  - 每日调仓基于动态择时信号 (dynamic_timing) + 动量轮动信号
  - 止盈止损: 单只ETF距成本 ±X% 触发
  - 全量 SQLite 持久化, 可长期追踪

用法:
  from engine.paper_trader import PaperTrader
  pt = PaperTrader()
  pt.create_account("模拟盘1", cash=100000)
  pt.rebalance("模拟盘1")     # 根据当前信号调仓
  pt.status("模拟盘1")        # 查看账户
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional

import pandas as pd
import numpy as np

from config import ROOT, COMMISSION, STAMP_DUTY, MIN_COMMISSION
from data.fetcher import load, fetch_etf

# ── 数据库路径 ──
DB_PATH = ROOT / "data" / "paper_trading.db"

# ── ETF 池 (与 main.py 保持一致) ──
ETF_POOL = ["510050", "510300", "510500", "159915", "588000",
            "512880", "512010", "512690", "510880", "511010"]

ETF_NAMES = {
    "510050": "上证50",   "510300": "沪深300",  "510500": "中证500",
    "159915": "创业板",    "588000": "科创50",   "512880": "证券",
    "512010": "医药",      "512690": "酒",       "510880": "红利",
    "511010": "国债",
}

# ── 默认参数 ──
DEFAULT_LOOKBACK = 10      # 动量窗口
DEFAULT_TOP_K = 3          # 持仓数
DEFAULT_REBAL_FREQ = 40    # 调仓频率(交易日)

# ── 风控 ──
STOP_LOSS_PCT = -0.08      # 单只 ETF 止损线 -8%
TAKE_PROFIT_PCT = 0.20     # 单只 ETF 止盈线 +20%


def _today_str() -> str:
    return date.today().isoformat()


class PaperTrader:
    """模拟盘交易引擎"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._migrate()

    # ════════════════════════════════════
    # 数据库
    # ════════════════════════════════════
    def _conn(self):
        return sqlite3.connect(str(self.db_path))

    def _migrate(self):
        """增量迁移 — 给旧数据库加新列"""
        with self._conn() as c:
            cur = c.execute("PRAGMA table_info(accounts)")
            cols = {r[1] for r in cur.fetchall()}
            if "total_cost" not in cols:
                c.execute("ALTER TABLE accounts ADD COLUMN total_cost REAL DEFAULT 0")

    def _init_db(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT DEFAULT 'sim',  -- sim / live
                    cash REAL NOT NULL,
                    total_cost REAL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    notes TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS positions (
                    account_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    shares INTEGER NOT NULL DEFAULT 0,
                    avg_cost REAL NOT NULL DEFAULT 0,
                    updated_at TEXT DEFAULT (datetime('now','localtime')),
                    PRIMARY KEY (account_id, symbol),
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,  -- buy / sell
                    shares INTEGER NOT NULL,
                    price REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    stamp_duty REAL DEFAULT 0,
                    total_value REAL NOT NULL,
                    signal TEXT DEFAULT '',
                    regime TEXT DEFAULT '',
                    position_pct REAL DEFAULT 0,
                    timestamp TEXT DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );
                CREATE TABLE IF NOT EXISTS nav_snapshots (
                    account_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    position_value REAL NOT NULL,
                    position_pct REAL NOT NULL,
                    regime TEXT DEFAULT '',
                    PRIMARY KEY (account_id, date),
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );
            """)

    # ════════════════════════════════════
    # 账户管理
    # ════════════════════════════════════
    def create_account(self, name: str, cash: float = 100_000,
                       acct_type: str = "sim", notes: str = "",
                       total_cost: float = None) -> int:
        """创建账户, 返回 account_id
        
        Args:
            name: 账户名
            cash: 初始现金
            acct_type: sim(模拟) / live(实盘)
            notes: 备注
            total_cost: 总成本(实盘已投入金额), 默认=cash
        """
        if total_cost is None:
            total_cost = cash
        with self._conn() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO accounts (name, type, cash, total_cost, notes) VALUES (?,?,?,?,?)",
                    (name, acct_type, cash, total_cost, notes))
                aid = cur.lastrowid
                print(f"✅ 账户已创建: [{aid}] {name} (现金¥{cash:,.0f} / 成本¥{total_cost:,.0f})")
                return aid
            except sqlite3.IntegrityError:
                print(f"❌ 账户名已存在: {name}")
                return -1

    def list_accounts(self) -> pd.DataFrame:
        """列出所有账户"""
        with self._conn() as c:
            df = pd.read_sql_query(
                "SELECT id, name, type, cash, total_cost, created_at, notes FROM accounts ORDER BY id",
                c)
        if df.empty:
            print("📭 暂无账户, 请先 create")
            return df

        # 计算当前总资产
        rows = []
        for _, r in df.iterrows():
            total = self._calc_total(r["id"], r["cash"])
            cost = r["total_cost"] or r["cash"]
            rows.append({
                "ID": r["id"], "名称": r["name"], "类型": r["type"],
                "现金": f"¥{r['cash']:,.0f}",
                "总资产": f"¥{total:,.0f}",
                "总成本": f"¥{cost:,.0f}",
                "净盈亏": f"¥{total-cost:+,.0f}",
                "创建": r["created_at"][:10], "备注": r["notes"] or "",
            })
        return pd.DataFrame(rows)

    def _get_account(self, name_or_id) -> Optional[dict]:
        """按名称或ID查账户"""
        with self._conn() as c:
            row = c.execute(
                "SELECT id, name, type, cash, total_cost, notes FROM accounts WHERE name=? OR id=?",
                (str(name_or_id), str(name_or_id))
            ).fetchone()
            if row:
                return {"id": row[0], "name": row[1], "type": row[2],
                        "cash": row[3], "total_cost": row[4] or row[3], "notes": row[5]}
        return None

    def _calc_total(self, account_id: int, cash: float) -> float:
        """计算账户总资产 (现金 + 持仓市值)"""
        positions = self._get_positions(account_id)
        if positions.empty:
            return cash
        total = cash
        for _, pos in positions.iterrows():
            price = self._latest_price(pos["symbol"])
            if price > 0:
                total += pos["shares"] * price
        return total

    def _get_positions(self, account_id: int) -> pd.DataFrame:
        """获取账户持仓"""
        with self._conn() as c:
            df = pd.read_sql_query(
                "SELECT symbol, shares, avg_cost FROM positions WHERE account_id=? AND shares>0",
                c, params=(account_id,))
        return df

    def _latest_price(self, symbol: str) -> float:
        """获取最新收盘价"""
        from config import DATA_MARKET
        path = DATA_MARKET / f"etf_{symbol}.parquet"
        if not path.exists():
            return 0
        df = load(path)
        if df.empty:
            return 0
        if "date" in df.columns:
            df = df.sort_values("date")
        close_col = "close" if "close" in df.columns else df.columns[-1]
        return float(df[close_col].iloc[-1])

    # ════════════════════════════════════
    # 信号生成 (复用现有引擎)
    # ════════════════════════════════════
    def _get_signals(self, top_k: int = DEFAULT_TOP_K,
                     lookback: int = DEFAULT_LOOKBACK) -> dict:
        """获取当前动量轮动 + 动态择时信号"""
        from engine.dynamic_timing import dynamic_position_now
        from live_signal import generate_signals

        sig = generate_signals(top_k=top_k, lookback=lookback)
        dp = dynamic_position_now()

        # extract top symbols from hold list [(code, {name, momentum, ...}), ...]
        hold = sig.get("hold", [])
        top_symbols = [h[0] for h in hold]
        top_names = [h[1]["name"] for h in hold]

        return {
            "top_symbols": top_symbols,
            "top_names": top_names,
            "hold": hold,
            "dynamic_pos": dp["position"],
            "regime": dp["regime"],
            "date": _today_str(),
        }

    # ════════════════════════════════════
    # 调仓执行
    # ════════════════════════════════════
    def rebalance(self, name_or_id, top_k: int = None,
                  lookback: int = None, dry_run: bool = False) -> dict:
        """根据当前信号调仓

        Args:
            name_or_id: 账户名或ID
            dry_run: True=只出建议不执行

        Returns:
            dict: 调仓报告
        """
        acct = self._get_account(name_or_id)
        if not acct:
            print(f"❌ 账户不存在: {name_or_id}")
            return {}

        if top_k is None:
            top_k = DEFAULT_TOP_K
        if lookback is None:
            lookback = DEFAULT_LOOKBACK

        # 1. 获取信号
        sig = self._get_signals(top_k=top_k, lookback=lookback)
        top_symbols = sig["top_symbols"]
        dynamic_pos = sig["dynamic_pos"]
        regime = sig["regime"]

        # 2. 获取当前持仓 & 价格
        positions = self._get_positions(acct["id"])
        held = set(positions["symbol"].tolist()) if not positions.empty else set()

        # 3. 获取最新价格
        prices = {}
        for s in ETF_POOL:
            p = self._latest_price(s)
            if p > 0:
                prices[s] = p
        for s in top_symbols:
            if s not in prices:
                prices[s] = self._latest_price(s)
        # 国债始终有价格
        if "511010" in prices:
            pass  # 防御锚

        # 4. 计算目标
        total_value = self._calc_total(acct["id"], acct["cash"])
        target_exposure = total_value * dynamic_pos          # 总风险敞口
        per_symbol_target = target_exposure / max(top_k, 1)   # 每只目标金额

        report = {
            "account": acct["name"],
            "date": _today_str(),
            "regime": regime,
            "dynamic_pos": f"{dynamic_pos*100:.0f}%",
            "total_value": f"¥{total_value:,.0f}",
            "target_exposure": f"¥{target_exposure:,.0f}",
            "cash_reserve": f"¥{total_value - target_exposure:,.0f}",
            "top_symbols": top_symbols,
            "actions": [],
            "alerts": [],
            "executed": not dry_run,
        }

        # 5. 止损检查 (先于调仓)
        if not positions.empty:
            for _, pos in positions.iterrows():
                sym = pos["symbol"]
                if sym not in prices or prices[sym] <= 0:
                    continue
                pnl_pct = (prices[sym] - pos["avg_cost"]) / pos["avg_cost"]
                if pnl_pct <= STOP_LOSS_PCT:
                    report["alerts"].append(
                        f"🔴 止损: {sym}({ETF_NAMES.get(sym,sym)}) 亏损 {pnl_pct*100:.1f}%")
                    if not dry_run:
                        self._execute_trade(acct["id"], sym, "sell",
                                           pos["shares"], prices[sym],
                                           signal="止损", regime=regime,
                                           position_pct=dynamic_pos)
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    report["alerts"].append(
                        f"🟢 止盈: {sym}({ETF_NAMES.get(sym,sym)}) 盈利 {pnl_pct*100:.1f}%")

        # 重新读持仓（止损后可能变了）
        positions = self._get_positions(acct["id"])
        held = set(positions["symbol"].tolist()) if not positions.empty else set()

        # 6. 卖出不在 Top K 的
        for sym in held:
            if sym not in top_symbols:
                pos_row = positions[positions["symbol"] == sym].iloc[0]
                price = prices.get(sym, 0)
                if price <= 0:
                    continue
                action = {
                    "action": "卖出",
                    "symbol": sym,
                    "name": ETF_NAMES.get(sym, sym),
                    "shares": int(pos_row["shares"]),
                    "price": price,
                    "reason": "不在TopK",
                }
                report["actions"].append(action)
                if not dry_run:
                    self._execute_trade(acct["id"], sym, "sell",
                                       int(pos_row["shares"]), price,
                                       signal="调仓-退出TopK", regime=regime,
                                       position_pct=dynamic_pos)

        # 7. 买入/加仓 Top K
        # 重新读持仓 + 现金
        positions = self._get_positions(acct["id"])
        acct = self._get_account(name_or_id)  # 刷新现金
        cash = acct["cash"]

        for sym in top_symbols:
            if sym not in prices or prices[sym] <= 0:
                continue
            price = prices[sym]
            current_shares = 0
            if not positions.empty and sym in positions["symbol"].values:
                current_shares = int(
                    positions[positions["symbol"] == sym]["shares"].iloc[0])

            target_value = per_symbol_target
            current_value = current_shares * price
            diff = target_value - current_value

            # 手续费预算
            est_commission = max(MIN_COMMISSION, abs(diff) * COMMISSION)
            if diff > est_commission + 100:  # 至少比手续费多100块才调
                shares_to_buy = int(diff // price // 100 * 100)  # 整手
                if shares_to_buy > 0:
                    cost = shares_to_buy * price + max(MIN_COMMISSION, shares_to_buy * price * COMMISSION)
                    if cost <= cash:
                        action = {
                            "action": "买入",
                            "symbol": sym,
                            "name": ETF_NAMES.get(sym, sym),
                            "shares": shares_to_buy,
                            "price": price,
                            "cost": f"¥{cost:,.0f}",
                            "reason": f"动量入选 Top{top_k}",
                        }
                        report["actions"].append(action)
                        if not dry_run:
                            self._execute_trade(acct["id"], sym, "buy",
                                               shares_to_buy, price,
                                               signal=f"调仓-入选Top{top_k}",
                                               regime=regime,
                                               position_pct=dynamic_pos)
            elif diff < -100:
                # 超配了，减仓
                shares_to_sell = int(-diff // price // 100 * 100)
                if shares_to_sell > 0 and shares_to_sell <= current_shares:
                    action = {
                        "action": "卖出(减仓)",
                        "symbol": sym,
                        "name": ETF_NAMES.get(sym, sym),
                        "shares": shares_to_sell,
                        "price": price,
                        "reason": "超配削减",
                    }
                    report["actions"].append(action)
                    if not dry_run:
                        self._execute_trade(acct["id"], sym, "sell",
                                           shares_to_sell, price,
                                           signal="调仓-超配削减", regime=regime,
                                           position_pct=dynamic_pos)

        # 8. 保存 NAV 快照
        if not dry_run:
            self._save_nav(acct["id"], regime, dynamic_pos)

        # 如果无操作
        if not report["actions"] and not report["alerts"]:
            report["summary"] = "✅ 持仓无需调整"

        return report

    def _execute_trade(self, account_id: int, symbol: str, action: str,
                       shares: int, price: float, signal: str = "",
                       regime: str = "", position_pct: float = 0):
        """执行单笔交易, 更新现金 & 持仓"""
        total = shares * price
        commission = max(MIN_COMMISSION, total * COMMISSION)
        stamp = total * STAMP_DUTY if action == "sell" else 0

        with self._conn() as c:
            # 更新现金
            if action == "buy":
                c.execute("UPDATE accounts SET cash = cash - ? WHERE id = ?",
                          (total + commission, account_id))
            else:
                c.execute("UPDATE accounts SET cash = cash + ? WHERE id = ?",
                          (total - commission - stamp, account_id))

            # 更新持仓
            existing = c.execute(
                "SELECT shares, avg_cost FROM positions WHERE account_id=? AND symbol=?",
                (account_id, symbol)).fetchone()

            if action == "buy":
                if existing:
                    new_shares = existing[0] + shares
                    new_cost = ((existing[0] * existing[1]) + (shares * price)) / new_shares
                    c.execute(
                        "UPDATE positions SET shares=?, avg_cost=?, updated_at=? WHERE account_id=? AND symbol=?",
                        (new_shares, new_cost, datetime.now().isoformat(), account_id, symbol))
                else:
                    c.execute(
                        "INSERT INTO positions (account_id, symbol, shares, avg_cost) VALUES (?,?,?,?)",
                        (account_id, symbol, shares, price))
            else:
                if existing:
                    new_shares = existing[0] - shares
                    if new_shares <= 0:
                        c.execute("DELETE FROM positions WHERE account_id=? AND symbol=?",
                                  (account_id, symbol))
                    else:
                        c.execute(
                            "UPDATE positions SET shares=?, updated_at=? WHERE account_id=? AND symbol=?",
                            (new_shares, datetime.now().isoformat(), account_id, symbol))

            # 记录交易
            c.execute(
                """INSERT INTO trades (account_id, symbol, action, shares, price,
                   commission, stamp_duty, total_value, signal, regime, position_pct)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (account_id, symbol, action, shares, price,
                 commission, stamp, total, signal, regime, position_pct))

    def _save_nav(self, account_id: int, regime: str, position_pct: float):
        """保存每日净值快照"""
        acct = self._get_account(str(account_id))
        if not acct:
            return
        total = self._calc_total(account_id, acct["cash"])
        pos_val = total - acct["cash"]

        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO nav_snapshots
                   (account_id, date, total_value, cash, position_value, position_pct, regime)
                   VALUES (?,?,?,?,?,?,?)""",
                (account_id, _today_str(), total, acct["cash"],
                 pos_val, position_pct, regime))

    # ════════════════════════════════════
    # 查询
    # ════════════════════════════════════
    def status(self, name_or_id) -> dict:
        """账户状态总览"""
        acct = self._get_account(name_or_id)
        if not acct:
            print(f"❌ 账户不存在: {name_or_id}")
            return {}

        positions = self._get_positions(acct["id"])
        total = self._calc_total(acct["id"], acct["cash"])
        pos_val = total - acct["cash"]
        cost = acct.get("total_cost", 100_000) or 100_000
        pnl = total - cost

        # 每只持仓明细
        holdings = []
        if not positions.empty:
            for _, pos in positions.iterrows():
                sym = pos["symbol"]
                price = self._latest_price(sym)
                mkt_val = pos["shares"] * price
                cost_val = pos["shares"] * pos["avg_cost"]
                pnl_pct = (price - pos["avg_cost"]) / pos["avg_cost"] * 100 if pos["avg_cost"] > 0 else 0
                holdings.append({
                    "symbol": sym,
                    "name": ETF_NAMES.get(sym, sym),
                    "shares": int(pos["shares"]),
                    "avg_cost": round(pos["avg_cost"], 4),
                    "price": round(price, 4),
                    "value": f"¥{mkt_val:,.0f}",
                    "pnl_pct": f"{pnl_pct:+.1f}%",
                })

        # 最近交易
        trades_df = self.trade_history(name_or_id, limit=5)

        # NAV 历史
        nav_df = self.nav_history(name_or_id)
        if not nav_df.empty:
            latest_nav = nav_df.iloc[-1]
            nav_return = (latest_nav["total_value"] / cost - 1) * 100
        else:
            nav_return = 0

        return {
            "account": acct,
            "total_value": total,
            "cash": acct["cash"],
            "total_cost": cost,
            "position_value": pos_val,
            "position_pct": f"{pos_val/total*100:.0f}%" if total > 0 else "0%",
            "pnl_total": f"¥{pnl:+,.0f}",
            "pnl_pct": f"{nav_return:+.1f}%",
            "holdings": holdings,
            "recent_trades": trades_df,
        }

    def trade_history(self, name_or_id, limit: int = 30) -> pd.DataFrame:
        """交易历史"""
        acct = self._get_account(name_or_id)
        if not acct:
            return pd.DataFrame()

        with self._conn() as c:
            df = pd.read_sql_query(
                """SELECT timestamp, action, symbol, shares, price,
                   commission, stamp_duty, total_value, signal, regime
                   FROM trades WHERE account_id=?
                   ORDER BY timestamp DESC LIMIT ?""",
                c, params=(acct["id"], limit))
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def nav_history(self, name_or_id) -> pd.DataFrame:
        """净值历史"""
        acct = self._get_account(name_or_id)
        if not acct:
            return pd.DataFrame()

        with self._conn() as c:
            df = pd.read_sql_query(
                """SELECT date, total_value, cash, position_value, position_pct, regime
                   FROM nav_snapshots WHERE account_id=?
                   ORDER BY date""",
                c, params=(acct["id"],))
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # ════════════════════════════════════
    # 多账户信号
    # ════════════════════════════════════
    def all_accounts_advice(self) -> pd.DataFrame:
        """所有账户的动态策略建议"""
        with self._conn() as c:
            df = pd.read_sql_query("SELECT id, name, type, cash FROM accounts", c)

        if df.empty:
            print("📭 无账户")
            return df

        sig = self._get_signals()
        dp = sig["dynamic_pos"]
        regime = sig["regime"]

        rows = []
        for _, r in df.iterrows():
            total = self._calc_total(r["id"], r["cash"])
            target = total * dp
            rows.append({
                "账户": r["name"],
                "类型": r["type"],
                "总资产": f"¥{total:,.0f}",
                "当前仓位": f"{dp*100:.0f}%",
                "建议敞口": f"¥{target:,.0f}",
                "市场": regime,
                "建议持仓": ", ".join(sig["top_names"][:3]),
            })
        return pd.DataFrame(rows)

    def reset_account(self, name_or_id, cash: float = 100_000):
        """重置账户 (清空持仓+交易记录, 恢复初始现金)"""
        acct = self._get_account(name_or_id)
        if not acct:
            print(f"❌ 账户不存在: {name_or_id}")
            return

        with self._conn() as c:
            c.execute("DELETE FROM positions WHERE account_id=?", (acct["id"],))
            c.execute("DELETE FROM trades WHERE account_id=?", (acct["id"],))
            c.execute("DELETE FROM nav_snapshots WHERE account_id=?", (acct["id"],))
            c.execute("UPDATE accounts SET cash=? WHERE id=?", (cash, acct["id"]))
        print(f"🔄 账户 [{acct['id']}] {acct['name']} 已重置, 现金=¥{cash:,.0f}")

    def delete_account(self, name_or_id):
        """删除账户及所有关联数据"""
        acct = self._get_account(name_or_id)
        if not acct:
            print(f"❌ 账户不存在: {name_or_id}")
            return

        with self._conn() as c:
            c.execute("DELETE FROM positions WHERE account_id=?", (acct["id"],))
            c.execute("DELETE FROM trades WHERE account_id=?", (acct["id"],))
            c.execute("DELETE FROM nav_snapshots WHERE account_id=?", (acct["id"],))
            c.execute("DELETE FROM accounts WHERE id=?", (acct["id"],))
        print(f"🗑️ 账户 [{acct['id']}] {acct['name']} 已删除")

    def update_cost_basis(self, name_or_id, new_cost: float):
        """更新账户总成本(实盘追加资金后调用)"""
        acct = self._get_account(name_or_id)
        if not acct:
            print(f"❌ 账户不存在: {name_or_id}")
            return

        old_cost = acct.get("total_cost", acct["cash"])
        with self._conn() as c:
            c.execute("UPDATE accounts SET total_cost=? WHERE id=?", (new_cost, acct["id"]))
        print(f"💰 [{acct['name']}] 总成本已更新: ¥{old_cost:,.0f} → ¥{new_cost:,.0f}")

    def manual_trade(self, name_or_id, symbol: str, action: str,
                     price: float, shares: int):
        """手动录入一笔实际交易(非策略驱动)
        
        Args:
            name_or_id: 账户名或ID
            symbol: 代码 (如 '510050')
            action: 'buy' 或 'sell'
            price: 成交价
            shares: 股数
        """
        acct = self._get_account(name_or_id)
        if not acct:
            print(f"❌ 账户不存在: {name_or_id}")
            return None

        if action not in ("buy", "sell"):
            print(f"❌ action 必须是 buy 或 sell")
            return None

        total = shares * price
        commission = max(MIN_COMMISSION, total * COMMISSION)
        stamp = total * STAMP_DUTY if action == "sell" else 0

        # 卖: 检查持仓够不够
        if action == "sell":
            pos = self._get_positions(acct["id"])
            held = 0
            if not pos.empty:
                match = pos[pos["symbol"] == symbol]
                if not match.empty:
                    held = int(match.iloc[0]["shares"])
            if shares > held:
                print(f"❌ 持仓不足: {symbol} 持有 {held} 股, 试图卖 {shares} 股")
                return None

        # 买: 检查现金够不够
        if action == "buy":
            cost = total + commission
            if cost > acct["cash"]:
                print(f"❌ 现金不足: ¥{acct['cash']:,.0f} < ¥{cost:,.0f}")
                return None

        # 获取动态择时 (标记用)
        regime = ""
        pos_pct = 0
        try:
            from engine.dynamic_timing import dynamic_position_now
            dp = dynamic_position_now()
            regime = dp.get("regime", "")
            pos_pct = dp.get("position", 0)
        except:
            pass

        # 执行交易
        self._execute_trade(acct["id"], symbol, action, shares, price,
                           signal="手动录入", regime=regime, position_pct=pos_pct)

        # 保存 NAV 快照
        self._save_nav(acct["id"], regime, pos_pct)

        act_cn = "买入" if action == "buy" else "卖出"
        name = ETF_NAMES.get(symbol, symbol)
        print(f"✅ [{acct['name']}] 手动{act_cn}: {symbol}({name}) {shares}股 @ ¥{price:.4f}  "
              f"金额¥{total:,.0f} 手续费¥{commission+stamp:,.0f}")

        return {
            "account": acct["name"],
            "action": act_cn,
            "symbol": symbol,
            "name": name,
            "shares": shares,
            "price": price,
            "total": total,
            "commission": commission + stamp,
        }
