# ===== LSTM模型模块 =====

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import os
import pickle

logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM预测模型"""
    
    def __init__(self, seq_length=60, lstm_units=[50, 50, 50], dropout_rate=0.2):
        """
        初始化LSTM模型
        
        Args:
            seq_length: 序列长度
            lstm_units: LSTM层单元数列表
            dropout_rate: Dropout比率
        """
        self.seq_length = seq_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        构建LSTM模型
        """
        model = Sequential()
        
        # 第一层LSTM
        model.add(LSTM(units=self.lstm_units[0], 
                      return_sequences=True, 
                      input_shape=(self.seq_length, 1)))
        model.add(Dropout(self.dropout_rate))
        
        # 中间LSTM层
        for units in self.lstm_units[1:-1]:
            model.add(LSTM(units=units, return_sequences=True))
            model.add(Dropout(self.dropout_rate))
        
        # 最后一层LSTM
        model.add(LSTM(units=self.lstm_units[-1]))
        model.add(Dropout(self.dropout_rate))
        
        # 输出层
        model.add(Dense(units=1))
        
        # 编译模型
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse',
                     metrics=['mae'])
        
        self.model = model
        logger.info("LSTM模型构建完成")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证���标
            epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            训练历史
        """
        if self.model is None:
            self.build_model()
        
        # 早停回调
        early_stop = EarlyStopping(monitor='val_loss', 
                                  patience=10, 
                                  restore_best_weights=True)
        
        logger.info("开始训练模型...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )
        
        logger.info("模型训练完成")
        return self.history
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 输入数据
            
        Returns:
            预测值
        """
        if self.model is None:
            raise ValueError("模型未构建")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        
        Args:
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            评估指标字典
        """
        predictions = self.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'mse': mean_squared_error(y_test, predictions)
        }
        
        logger.info(f"模型评估: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
        
        return metrics
    
    def save_model(self, filepath):
        """
        保存模型
        """
        if self.model is None:
            raise ValueError("模型未构建")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"模型已保存: {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        """
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        logger.info(f"模型已加载: {filepath}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # 测试
    from data_fetcher import DataFetcher
    from data_preprocessor import DataPreprocessor
    
    fetcher = DataFetcher('000960.SZ')
    data = fetcher.fetch_data(start_date='2023-01-01', end_date='2026-06-13')
    
    preprocessor = DataPreprocessor()
    prices = data['Close'].values.reshape(-1, 1)
    normalized_prices = preprocessor.normalize(prices)
    
    X, y = preprocessor.create_sequences(normalized_prices, seq_length=60)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # 创建模型
    lstm = LSTMModel(seq_length=60)
    lstm.build_model()
    
    # 训练
    lstm.train(X_train, y_train, X_test, y_test, epochs=10)
    
    # 评估
    metrics = lstm.evaluate(X_test, y_test)
    print(f"评估指标: {metrics}")
