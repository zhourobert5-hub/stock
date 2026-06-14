# ===== 融合模型模块 =====

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

class EnsembleModel:
    """LSTM + Prophet融合模型"""
    
    def __init__(self, lstm_model, prophet_model, lstm_weight=0.7, prophet_weight=0.3):
        """
        初始化融合模型
        
        Args:
            lstm_model: LSTM模型实例
            prophet_model: Prophet模型实例
            lstm_weight: LSTM权重
            prophet_weight: Prophet权重
        """
        self.lstm_model = lstm_model
        self.prophet_model = prophet_model
        self.lstm_weight = lstm_weight
        self.prophet_weight = prophet_weight
        
        # 验证权重
        total_weight = lstm_weight + prophet_weight
        self.lstm_weight = lstm_weight / total_weight
        self.prophet_weight = prophet_weight / total_weight
        
        logger.info(f"融合模型权重 - LSTM: {self.lstm_weight:.2%}, Prophet: {self.prophet_weight:.2%}")
    
    def predict_lstm(self, X):
        """
        LSTM预测
        
        Args:
            X: LSTM输入数据
            
        Returns:
            预测值
        """
        return self.lstm_model.predict(X)
    
    def predict_prophet(self, df, periods=5):
        """
        Prophet预测
        
        Args:
            df: Prophet输入数据
            periods: 预测周期
            
        Returns:
            预测值
        """
        forecast = self.prophet_model.forecast(periods=periods)
        return forecast['yhat'].values
    
    def predict_ensemble(self, lstm_pred, prophet_pred):
        """
        融合预测
        
        Args:
            lstm_pred: LSTM预测值
            prophet_pred: Prophet预测值
            
        Returns:
            融合预测值
        """
        # 确保长度一致
        if len(lstm_pred) != len(prophet_pred):
            min_len = min(len(lstm_pred), len(prophet_pred))
            lstm_pred = lstm_pred[:min_len]
            prophet_pred = prophet_pred[:min_len]
        
        # 加权平均
        ensemble_pred = (self.lstm_weight * lstm_pred.flatten() + 
                        self.prophet_weight * prophet_pred.flatten())
        
        return ensemble_pred
    
    def evaluate(self, lstm_pred, prophet_pred, y_true):
        """
        评估融合模型
        
        Args:
            lstm_pred: LSTM预测值
            prophet_pred: Prophet预测值
            y_true: 真实值
            
        Returns:
            各模型的评估指标
        """
        # 融合预测
        ensemble_pred = self.predict_ensemble(lstm_pred, prophet_pred)
        
        # 确保长度一致
        min_len = min(len(ensemble_pred), len(y_true))
        ensemble_pred = ensemble_pred[:min_len]
        y_true = y_true[:min_len]
        
        # 计算指标
        lstm_rmse = np.sqrt(mean_squared_error(y_true, lstm_pred[:min_len].flatten()))
        prophet_rmse = np.sqrt(mean_squared_error(y_true, prophet_pred[:min_len].flatten()))
        ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
        
        lstm_mae = mean_absolute_error(y_true, lstm_pred[:min_len].flatten())
        prophet_mae = mean_absolute_error(y_true, prophet_pred[:min_len].flatten())
        ensemble_mae = mean_absolute_error(y_true, ensemble_pred)
        
        metrics = {
            'lstm': {'rmse': lstm_rmse, 'mae': lstm_mae},
            'prophet': {'rmse': prophet_rmse, 'mae': prophet_mae},
            'ensemble': {'rmse': ensemble_rmse, 'mae': ensemble_mae}
        }
        
        logger.info(f"LSTM - RMSE: {lstm_rmse:.4f}, MAE: {lstm_mae:.4f}")
        logger.info(f"Prophet - RMSE: {prophet_rmse:.4f}, MAE: {prophet_mae:.4f}")
        logger.info(f"融合 - RMSE: {ensemble_rmse:.4f}, MAE: {ensemble_mae:.4f}")
        
        return metrics
    
    def get_confidence(self, ensemble_pred, lstm_pred, prophet_pred):
        """
        计算预测置信度
        
        Args:
            ensemble_pred: 融合预测值
            lstm_pred: LSTM预测值
            prophet_pred: Prophet预测值
            
        Returns:
            置信度（0-1之间）
        """
        # 计算三者的一致性（使用相对标准差）
        predictions = np.array([lstm_pred.flatten(), prophet_pred.flatten(), ensemble_pred])
        
        # 计算各预测值的相对偏差
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 避免除以零
        std_pred = np.where(std_pred == 0, 1e-10, std_pred)
        relative_std = std_pred / np.abs(mean_pred)
        
        # 置信度 = 1 - 相对标准差（限制在0-1之间）
        confidence = 1 - np.clip(relative_std, 0, 1)
        
        return np.mean(confidence)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("融合模型模块已准备好")
