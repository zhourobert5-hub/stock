#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
锡业股份智能预测系统 - 主程序

功能:
1. 数据获取与预处理
2. LSTM + Prophet 融合预测
3. 历史回测
4. 实时交易机器人
"""

import sys
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

# 导入配置
import config

# 导入核心模块
from src.data_fetcher import DataFetcher
from src.data_preprocessor import DataPreprocessor
from src.lstm_model import LSTMModel
from src.prophet_model import ProphetModel
from src.ensemble_model import EnsembleModel
from src.backtester import Backtester
from src.trader_bot import TraderBot

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format=config.LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(config.LOGGING_CONFIG['file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StockPredictionSystem:
    """锡业股份预测系统主类"""
    
    def __init__(self):
        """初始化系统"""
        logger.info(f"=" * 60)
        logger.info(f"{config.PROJECT_NAME} v{config.VERSION}")
        logger.info(f"Created: {config.CREATED_DATE}")
        logger.info(f"Stock: {config.STOCK_NAME} ({config.STOCK_TICKER})")
        logger.info(f"=" * 60)
        
        self.fetcher = None
        self.preprocessor = None
        self.lstm_model = None
        self.prophet_model = None
        self.ensemble_model = None
        self.backtester = None
        self.trader_bot = None
        self.data = None
        
    def initialize(self):
        """初始化所有组件"""
        logger.info("\n初始化系统组件...")
        
        # 初始化数据获取器
        self.fetcher = DataFetcher(
            config.STOCK_TICKER,
            cache_dir=config.DATA_DIR
        )
        logger.info("✓ 数据获取器初始化完成")
        
        # 初始化数据预处理器
        self.preprocessor = DataPreprocessor(
            feature_range=config.LSTM_CONFIG['feature_scale']
        )
        logger.info("✓ 数据预处理器初始化完成")
        
        # 初始化LSTM模型
        self.lstm_model = LSTMModel(
            seq_length=config.LSTM_CONFIG['seq_length'],
            lstm_units=config.LSTM_CONFIG['lstm_units'],
            dropout_rate=config.LSTM_CONFIG['dropout_rate']
        )
        logger.info("✓ LSTM模型初始化完成")
        
        # 初始化Prophet模型
        self.prophet_model = ProphetModel(
            yearly_seasonality=config.PROPHET_CONFIG['yearly_seasonality'],
            weekly_seasonality=config.PROPHET_CONFIG['weekly_seasonality'],
            daily_seasonality=config.PROPHET_CONFIG['daily_seasonality'],
            seasonality_mode=config.PROPHET_CONFIG['seasonality_mode']
        )
        logger.info("✓ Prophet模型初始化完成")
        
        # 初始化回测引擎
        self.backtester = Backtester(
            initial_capital=config.BACKTEST_CONFIG['initial_capital'],
            position_size=config.BACKTEST_CONFIG['position_size'],
            stop_loss=config.BACKTEST_CONFIG['stop_loss'],
            take_profit=config.BACKTEST_CONFIG['take_profit'],
            commission=config.BACKTEST_CONFIG['commission'],
            slippage=config.BACKTEST_CONFIG['slippage']
        )
        logger.info("✓ 回测引擎初始化完成")
        
    def fetch_and_prepare_data(self):
        """获取并准备数据"""
        logger.info("\n获取数据...")
        
        # 获取历史数据
        end_date = datetime.now().strftime('%Y-%m-%d')
        self.data = self.fetcher.fetch_data(
            start_date=config.STOCK_START_DATE,
            end_date=end_date,
            use_cache=True
        )
        logger.info(f"✓ 获取 {len(self.data)} 条数据")
        
        # 添加技术指标
        self.data = self.fetcher.add_technical_indicators(self.data)
        logger.info(f"✓ 添加技术指标")
        
        return self.data
    
    def prepare_lstm_data(self):
        """准备LSTM数据"""
        logger.info("\n准备LSTM数据...")
        
        # 提取收盘价
        prices = self.data['Close'].values.reshape(-1, 1)
        
        # 归一化
        normalized_prices = self.preprocessor.normalize(prices)
        
        # 创建序列
        X, y = self.preprocessor.create_sequences(
            normalized_prices,
            seq_length=config.LSTM_CONFIG['seq_length']
        )
        logger.info(f"✓ 序列数据形状: X={X.shape}, y={y.shape}")
        
        # 分割数据
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            X, y,
            train_ratio=config.LSTM_CONFIG['train_ratio']
        )
        
        return X_train, X_test, y_train, y_test, prices
    
    def train_lstm(self, X_train, X_test, y_train, y_test):
        """训练LSTM模型"""
        logger.info("\n训练LSTM模型...")
        
        # 构建模型
        self.lstm_model.build_model()
        
        # 训练
        history = self.lstm_model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=config.LSTM_CONFIG['epochs'],
            batch_size=config.LSTM_CONFIG['batch_size']
        )
        
        # 评估
        metrics = self.lstm_model.evaluate(X_test, y_test)
        logger.info(f"✓ LSTM评估指标: {metrics}")
        
        # 保存模型
        model_path = f"{config.MODELS_DIR}/lstm_model.h5"
        self.lstm_model.save_model(model_path)
        
        return metrics
    
    def train_prophet(self):
        """训练Prophet模型"""
        logger.info("\n训练Prophet模型...")
        
        # 准备数据
        df = self.prophet_model.prepare_data(
            self.data['Close'],
            self.data.index
        )
        
        # 分割数据
        train_size = int(len(df) * 0.8)
        df_train = df[:train_size]
        df_test = df[train_size:]
        
        # 训练
        self.prophet_model.train(df_train)
        
        # 评估
        metrics, forecast_eval = self.prophet_model.evaluate(df_test)
        logger.info(f"✓ Prophet评估指标: {metrics}")
        
        return df, metrics
    
    def create_ensemble_model(self):
        """创建融合模型"""
        logger.info("\n创建融合模型...")
        
        self.ensemble_model = EnsembleModel(
            lstm_model=self.lstm_model,
            prophet_model=self.prophet_model,
            lstm_weight=config.ENSEMBLE_CONFIG['lstm_weight'],
            prophet_weight=config.ENSEMBLE_CONFIG['prophet_weight']
        )
        logger.info("✓ 融合模型创建完成")
    
    def run_backtest(self, X_test, y_test, prices):
        """运行回测"""
        logger.info("\n运行回测...")
        
        # 获取预测值
        lstm_pred = self.lstm_model.predict(X_test)
        
        # 反归一化
        lstm_pred_real = self.preprocessor.denormalize(lstm_pred)
        y_test_real = self.preprocessor.denormalize(y_test)
        
        # 获取Prophet预测
        df = self.prophet_model.prepare_data(
            self.data['Close'],
            self.data.index
        )
        train_size = int(len(df) * 0.8)
        df_test = df[train_size:]
        
        forecast = self.prophet_model.model.predict(df_test[['ds']])
        prophet_pred_real = forecast['yhat'].values[-len(lstm_pred_real):]
        
        # 融合预测
        ensemble_pred = self.ensemble_model.predict_ensemble(
            lstm_pred_real,
            prophet_pred_real.reshape(-1, 1)
        )
        
        # 获取置信度
        confidence = self.ensemble_model.get_confidence(
            ensemble_pred,
            lstm_pred_real,
            prophet_pred_real
        )
        
        logger.info(f"✓ 平均置信度: {confidence:.2%}")
        
        # 运行回测
        metrics = self.trader_bot.run_backtest(
            self.data,
            lstm_pred_real,
            prophet_pred_real.reshape(-1, 1)
        )
        
        # 输出回测结果
        self.backtester.print_summary()
        
        return metrics
    
    def generate_forecast(self, days=5):
        """生成未来预测"""
        logger.info(f"\n生成未来{days}天预测...")
        
        # Prophet预测
        prophet_forecast = self.prophet_model.forecast(periods=days)
        logger.info("\nProphet预测结果:")
        print(prophet_forecast)
        
        return prophet_forecast
    
    def run_full_pipeline(self):
        """运行完整流程"""
        try:
            logger.info("\n" + "="*60)
            logger.info("开始运行完整流程")
            logger.info("="*60)
            
            # 1. 初始化
            self.initialize()
            
            # 2. 获取数据
            self.fetch_and_prepare_data()
            
            # 3. 准备LSTM数据
            X_train, X_test, y_train, y_test, prices = self.prepare_lstm_data()
            
            # 4. 训练LSTM
            lstm_metrics = self.train_lstm(X_train, X_test, y_train, y_test)
            
            # 5. 训练Prophet
            df, prophet_metrics = self.train_prophet()
            
            # 6. 创建融合模型
            self.create_ensemble_model()
            
            # 7. 初始化交易机器人
            self.trader_bot = TraderBot(
                self.ensemble_model,
                self.fetcher,
                self.backtester,
                prediction_threshold=config.TRADER_BOT_CONFIG['prediction_threshold'],
                check_interval=config.TRADER_BOT_CONFIG['check_interval']
            )
            logger.info("✓ 交易机器人初始化完成")
            
            # 8. 运行回测
            backtest_metrics = self.run_backtest(X_test, y_test, prices)
            
            # 9. 生成未来预测
            forecast = self.generate_forecast(
                days=config.PREDICTION_CONFIG['forecast_days']
            )
            
            logger.info("\n" + "="*60)
            logger.info("流程完成！")
            logger.info("="*60)
            
            return {
                'lstm_metrics': lstm_metrics,
                'prophet_metrics': prophet_metrics,
                'backtest_metrics': backtest_metrics,
                'forecast': forecast
            }
            
        except Exception as e:
            logger.error(f"系统错误: {e}", exc_info=True)
            return None
    
    def start_trader_bot(self):
        """启动交易机器人"""
        logger.info("\n启动交易机器人...")
        
        if self.trader_bot is None:
            logger.error("交易机器人未初始化")
            return
        
        self.trader_bot.start()
        logger.info("✓ 交易机器人已启动")
        logger.info("按 Ctrl+C 停止")
        
        try:
            while True:
                status = self.trader_bot.get_status()
                logger.info(f"机器人状态: {status}")
                import time
                time.sleep(60)
        except KeyboardInterrupt:
            self.trader_bot.stop()
            logger.info("交易机器人已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='锡业股份智能预测系统')
    parser.add_argument(
        '--mode',
        choices=['train', 'backtest', 'forecast', 'bot'],
        default='train',
        help='运行模式 (default: train)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=5,
        help='预测天数 (default: 5)'
    )
    
    args = parser.parse_args()
    
    # 创建系统实例
    system = StockPredictionSystem()
    
    if args.mode == 'train':
        # 训练模式
        results = system.run_full_pipeline()
    
    elif args.mode == 'backtest':
        # 回测模式
        system.initialize()
        system.fetch_and_prepare_data()
        X_train, X_test, y_train, y_test, prices = system.prepare_lstm_data()
        system.train_lstm(X_train, X_test, y_train, y_test)
        system.train_prophet()
        system.create_ensemble_model()
        system.trader_bot = TraderBot(
            system.ensemble_model,
            system.fetcher,
            system.backtester
        )
        system.run_backtest(X_test, y_test, prices)
    
    elif args.mode == 'forecast':
        # 预测模式
        system.initialize()
        system.fetch_and_prepare_data()
        X_train, X_test, y_train, y_test, _ = system.prepare_lstm_data()
        system.train_lstm(X_train, X_test, y_train, y_test)
        system.train_prophet()
        system.create_ensemble_model()
        system.generate_forecast(days=args.days)
    
    elif args.mode == 'bot':
        # 交易机器人模式
        system.run_full_pipeline()
        system.start_trader_bot()


if __name__ == '__main__':
    main()
