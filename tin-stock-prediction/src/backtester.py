# ===== 回测引擎模块 =====

import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Backtester:
    """回测引擎"""
    
    def __init__(self, initial_capital=100000, position_size=0.1, 
                 stop_loss=0.05, take_profit=0.15, commission=0.001, slippage=0.001):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            position_size: 单笔仓位比例
            stop_loss: 止损比例
            take_profit: 止盈比例
            commission: 交易佣金
            slippage: 滑点
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.commission = commission
        self.slippage = slippage
        
        # 回测结果
        self.cash = initial_capital
        self.positions = []  # 持仓列表
        self.trades = []     # 交易记录
        self.equity_curve = []  # 账户资产曲线
        
    def calculate_trade_quantity(self, price):
        """
        计算交易数量
        
        Args:
            price: 当前价格
            
        Returns:
            交易数量
        """
        trade_amount = self.initial_capital * self.position_size
        quantity = int(trade_amount / price)
        return quantity
    
    def open_position(self, date, price, signal, prediction_confidence=0.5):
        """
        开仓
        
        Args:
            date: 开仓日期
            price: 开仓价格
            signal: 交易信号 (1=做多, -1=做空)
            prediction_confidence: 预测置信度
            
        Returns:
            是否成功开仓
        """
        if self.cash <= 0:
            return False
        
        quantity = self.calculate_trade_quantity(price)
        if quantity <= 0:
            return False
        
        # 考虑滑点
        actual_price = price * (1 + self.slippage * signal)
        cost = quantity * actual_price * (1 + self.commission)
        
        if cost > self.cash:
            return False
        
        position = {
            'date': date,
            'entry_price': actual_price,
            'quantity': quantity,
            'signal': signal,
            'confidence': prediction_confidence,
            'exit_price': None,
            'exit_date': None,
            'pnl': 0,
            'return': 0
        }
        
        self.positions.append(position)
        self.cash -= cost
        
        logger.info(f"开仓: {date} | 价格: {actual_price:.2f} | 数量: {quantity} | 信号: {'做多' if signal > 0 else '做空'}")
        
        return True
    
    def close_position(self, position_index, date, price, reason='manual'):
        """
        平仓
        
        Args:
            position_index: 持仓索引
            date: 平仓日期
            price: 平仓价格
            reason: 平仓原因 ('manual', 'stop_loss', 'take_profit')
            
        Returns:
            平仓收益
        """
        if position_index >= len(self.positions):
            return 0
        
        position = self.positions[position_index]
        
        # 考虑滑点
        actual_price = price * (1 - self.slippage * position['signal'])
        revenue = position['quantity'] * actual_price * (1 - self.commission)
        
        # 计算盈亏
        pnl = revenue - (position['quantity'] * position['entry_price'] * (1 + self.commission))
        pnl_return = (actual_price - position['entry_price']) / position['entry_price']
        
        # 更新持仓
        position['exit_price'] = actual_price
        position['exit_date'] = date
        position['pnl'] = pnl
        position['return'] = pnl_return
        
        # 更新现金
        self.cash += revenue
        
        # 记录交易
        trade = {
            'entry_date': position['date'],
            'exit_date': date,
            'entry_price': position['entry_price'],
            'exit_price': actual_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'return': pnl_return,
            'reason': reason
        }
        self.trades.append(trade)
        
        logger.info(f"平仓: {date} | 价格: {actual_price:.2f} | 收益: {pnl:.2f} | 原因: {reason}")
        
        return pnl
    
    def check_stop_loss_and_take_profit(self, date, current_price):
        """
        检查止损和止盈
        
        Args:
            date: 当前日期
            current_price: 当前价格
        """
        positions_to_close = []
        
        for i, position in enumerate(self.positions):
            if position['exit_price'] is not None:
                continue  # 已平仓
            
            entry_price = position['entry_price']
            price_change = (current_price - entry_price) / entry_price
            
            # 检查止损
            if price_change < -self.stop_loss:
                positions_to_close.append((i, 'stop_loss'))
            # 检查止盈
            elif price_change > self.take_profit:
                positions_to_close.append((i, 'take_profit'))
        
        # 平仓
        for pos_idx, reason in positions_to_close:
            self.close_position(pos_idx, date, current_price, reason)
    
    def update_equity(self, date, current_price):
        """
        更新账户资产
        
        Args:
            date: 当前日期
            current_price: 当前价格
        """
        # 计算未实现盈亏
        unrealized_pnl = 0
        for position in self.positions:
            if position['exit_price'] is None:
                unrealized_pnl += position['quantity'] * (current_price - position['entry_price'])
        
        total_equity = self.cash + unrealized_pnl
        self.equity_curve.append({
            'date': date,
            'equity': total_equity,
            'cash': self.cash,
            'unrealized_pnl': unrealized_pnl
        })
    
    def get_performance_metrics(self):
        """
        计算回测性能指标
        
        Returns:
            性能指标字典
        """
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        # 基本指标
        total_return = (self.cash - self.initial_capital) / self.initial_capital
        total_trades = len(self.trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 收益指标
        avg_profit = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_profit * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else 0
        
        # 风险指标
        equity_curve_df = pd.DataFrame(self.equity_curve)
        if len(equity_curve_df) > 0:
            returns = equity_curve_df['equity'].pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_drawdown = self._calculate_max_drawdown(equity_curve_df['equity'].values)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        metrics = {
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_equity': self.cash
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, equity_values):
        """
        计算最大回撤
        
        Args:
            equity_values: 资产值列表
            
        Returns:
            最大回撤
        """
        max_equity = equity_values[0]
        max_drawdown = 0
        
        for equity in equity_values:
            if equity > max_equity:
                max_equity = equity
            drawdown = (max_equity - equity) / max_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def print_summary(self):
        """
        打印回测总结
        """
        metrics = self.get_performance_metrics()
        
        print("\n" + "="*50)
        print("回测总结")
        print("="*50)
        print(f"初始资金: {self.initial_capital:,.2f}")
        print(f"最终资金: {self.cash:,.2f}")
        print(f"总收益: {metrics.get('total_return', 0):.2%}")
        print(f"总交易数: {metrics.get('total_trades', 0)}")
        print(f"盈利交易: {metrics.get('winning_trades', 0)}")
        print(f"亏损交易: {metrics.get('losing_trades', 0)}")
        print(f"胜率: {metrics.get('win_rate', 0):.2%}")
        print(f"利润因子: {metrics.get('profit_factor', 0):.2f}")
        print(f"Sharpe比率: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        print("="*50)
