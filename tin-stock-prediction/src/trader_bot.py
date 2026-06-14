# ===== 自动化交易机器人 =====

import numpy as np
import pandas as pd
import logging
from datetime import datetime, time
import schedule
import threading

logger = logging.getLogger(__name__)

class TraderBot:
    """自动化交易机器人"""
    
    def __init__(self, ensemble_model, data_fetcher, backtester, 
                 prediction_threshold=0.55, check_interval=300):
        """
        初始化交易机器人
        
        Args:
            ensemble_model: 融合预测模型
            data_fetcher: 数据获取器
            backtester: 回测引擎
            prediction_threshold: 预测置信度阈值
            check_interval: 检查间隔（秒）
        """
        self.ensemble_model = ensemble_model
        self.data_fetcher = data_fetcher
        self.backtester = backtester
        self.prediction_threshold = prediction_threshold
        self.check_interval = check_interval
        self.running = False
        
    def generate_signal(self, prediction, confidence):
        """
        生成交易信号
        
        Args:
            prediction: 预测的价格变化方向
            confidence: 预测置信度
            
        Returns:
            信号 (1=做多, -1=做空, 0=无信号)
        """
        if confidence < self.prediction_threshold:
            return 0
        
        if prediction > 0:
            return 1  # 做多
        else:
            return -1  # 做空
    
    def check_market_hours(self):
        """
        检查是否在交易时间内
        
        Returns:
            True/False
        """
        now = datetime.now()
        
        # 中国A股交易时间: 9:30-11:30, 13:00-15:00 (周一到周五)
        if now.weekday() >= 5:  # 周末
            return False
        
        current_time = now.time()
        
        if (time(9, 30) <= current_time <= time(11, 30) or 
            time(13, 0) <= current_time <= time(15, 0)):
            return True
        
        return False
    
    def execute_trade(self, signal, current_price, current_date, confidence):
        """
        执行交易
        
        Args:
            signal: 交易信号
            current_price: 当前价格
            current_date: 当前日期
            confidence: 置信度
            
        Returns:
            是否成功执行
        """
        if signal == 0:
            return False
        
        # 检查是否已有相反方向持仓
        for position in self.backtester.positions:
            if position['exit_price'] is None and position['signal'] != signal:
                # 平掉相反方向的持仓
                idx = self.backtester.positions.index(position)
                self.backtester.close_position(idx, current_date, current_price, 'reverse')
        
        # 开新仓
        return self.backtester.open_position(current_date, current_price, signal, confidence)
    
    def run_backtest(self, data, lstm_predictions, prophet_predictions):
        """
        运行回测
        
        Args:
            data: 历史数据DataFrame
            lstm_predictions: LSTM预测值
            prophet_predictions: Prophet预测值
            
        Returns:
            回测结果
        """
        logger.info("开始运行回测...")
        
        for i in range(len(data) - 1):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]
            next_price = data['Close'].iloc[i + 1]
            
            # 获取预测
            if i < len(lstm_predictions) and i < len(prophet_predictions):
                lstm_pred = lstm_predictions[i]
                prophet_pred = prophet_predictions[i]
                
                # 融合预测
                ensemble_pred = self.ensemble_model.predict_ensemble(
                    np.array([[lstm_pred]]), 
                    np.array([prophet_pred])
                )[0]
                
                # 获取置信度
                confidence = self.ensemble_model.get_confidence(
                    np.array([ensemble_pred]),
                    np.array([[lstm_pred]]),
                    np.array([prophet_pred])
                )
                
                # 生成信号
                signal = self.generate_signal(ensemble_pred - current_price, confidence)
                
                # 执行交易
                if signal != 0:
                    self.execute_trade(signal, current_price, current_date, confidence)
            
            # 检查止损和止盈
            self.backtester.check_stop_loss_and_take_profit(current_date, next_price)
            
            # 更新账户资产
            self.backtester.update_equity(current_date, next_price)
        
        logger.info("回测完成")
        return self.backtester.get_performance_metrics()
    
    def monitor_market(self):
        """
        实时监控市场
        """
        logger.info("市场监控启动")
        
        while self.running:
            try:
                # 检查交易时间
                if not self.check_market_hours():
                    logger.debug("非交易时间，暂停")
                    continue
                
                # 获取最新数据
                current_price = self.data_fetcher.get_latest_price()
                current_date = datetime.now()
                
                # 获取预测
                logger.info(f"获取预测... (当前价格: {current_price:.2f})")
                
                # 这里应该集成实时预测逻辑
                # 暂时略过，等待集成完整系统
                
            except Exception as e:
                logger.error(f"监控出错: {e}")
            
            # 等待下次检查
            threading.Event().wait(self.check_interval)
    
    def start(self):
        """
        启动交易机器人
        """
        if self.running:
            logger.warning("机器人已在运行")
            return
        
        self.running = True
        logger.info("交易机器人启动")
        
        # 在后台线程运行监控
        monitor_thread = threading.Thread(target=self.monitor_market, daemon=True)
        monitor_thread.start()
    
    def stop(self):
        """
        停止交易机器人
        """
        self.running = False
        logger.info("交易机器人停止")
    
    def get_status(self):
        """
        获取机器人状态
        
        Returns:
            状态字典
        """
        return {
            'running': self.running,
            'positions': len(self.backtester.positions),
            'cash': self.backtester.cash,
            'prediction_threshold': self.prediction_threshold
        }
