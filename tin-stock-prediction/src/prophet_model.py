# ===== Prophet模型模块 =====

import pandas as pd
import numpy as np
from prophet import Prophet
import logging
import os

logger = logging.getLogger(__name__)

class ProphetModel:
    """Prophet时间序列预测模型"""
    
    def __init__(self, yearly_seasonality=True, weekly_seasonality=True, 
                 daily_seasonality=False, seasonality_mode='additive'):
        """
        初始化Prophet模型
        
        Args:
            yearly_seasonality: 年度季节性
            weekly_seasonality: 周度季节性
            daily_seasonality: 日度季节性
            seasonality_mode: 季节性模式 ('additive' or 'multiplicative')
        """
        self.model = None
        self.config = {
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'seasonality_mode': seasonality_mode
        }
    
    def prepare_data(self, prices_df, date_index):
        """
        准备Prophet所需的数据格式
        
        Args:
            prices_df: 价格Series
            date_index: 日期索引
            
        Returns:
            DataFrame: 包含ds和y列的数据框
        """
        df = pd.DataFrame({
            'ds': date_index,
            'y': prices_df.values
        })
        
        logger.info(f"数据准备完成: {df.shape}")
        return df
    
    def train(self, df):
        """
        训练Prophet模型
        
        Args:
            df: 包含ds和y列的DataFrame
        """
        logger.info("开始训练Prophet模型...")
        
        self.model = Prophet(
            yearly_seasonality=self.config['yearly_seasonality'],
            weekly_seasonality=self.config['weekly_seasonality'],
            daily_seasonality=self.config['daily_seasonality'],
            seasonality_mode=self.config['seasonality_mode'],
            interval_width=0.95
        )
        
        self.model.fit(df)
        logger.info("Prophet模型训练完成")
    
    def forecast(self, periods=5):
        """
        预测未来价格
        
        Args:
            periods: 预测天数
            
        Returns:
            DataFrame: 预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        logger.info(f"预测完成: {periods}天")
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    
    def evaluate(self, df_test):
        """
        评估模型
        
        Args:
            df_test: 测试数据
            
        Returns:
            评估指标
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        forecast = self.model.predict(df_test[['ds']])
        
        rmse = np.sqrt(mean_squared_error(df_test['y'], forecast['yhat']))
        mae = mean_absolute_error(df_test['y'], forecast['yhat'])
        mape = np.mean(np.abs((df_test['y'].values - forecast['yhat'].values) / 
                             df_test['y'].values)) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'mse': mean_squared_error(df_test['y'], forecast['yhat'])
        }
        
        logger.info(f"模型评估: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
        
        return metrics, forecast
    
    def plot_forecast(self, forecast):
        """
        绘制预测结果
        
        Args:
            forecast: Prophet预测结果
        """
        try:
            import matplotlib.pyplot as plt
            
            fig = self.model.plot(forecast)
            plt.title('Prophet预测结果')
            plt.xlabel('日期')
            plt.ylabel('价格')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"绘制失败: {e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # 测试
    from data_fetcher import DataFetcher
    
    fetcher = DataFetcher('000960.SZ')
    data = fetcher.fetch_data(start_date='2023-01-01', end_date='2026-06-13')
    
    # 准备数据
    prophet = ProphetModel()
    df = prophet.prepare_data(data['Close'], data.index)
    
    # 分割数据
    train_size = int(len(df) * 0.8)
    df_train = df[:train_size]
    df_test = df[train_size:]
    
    # 训练
    prophet.train(df_train)
    
    # 评估
    metrics, forecast_eval = prophet.evaluate(df_test)
    print(f"评估指标: {metrics}")
    
    # 预测未来5天
    forecast = prophet.forecast(periods=5)
    print("\n未来5天预测:")
    print(forecast)
