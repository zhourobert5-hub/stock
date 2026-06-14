# ===== 数据预处理模块 =====

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理"""
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.original_shape = None
        
    def normalize(self, data):
        """
        数据归一化
        
        Args:
            data: numpy array or DataFrame
            
        Returns:
            归一化后的数据
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.original_shape = data.shape
        return self.scaler.fit_transform(data)
    
    def denormalize(self, data):
        """
        反归一化
        
        Args:
            data: 归一化的数据
            
        Returns:
            原始数据
        """
        return self.scaler.inverse_transform(data)
    
    def create_sequences(self, data, seq_length=60):
        """
        创建序列数据（滑动窗口）
        
        Args:
            data: 输入数据
            seq_length: 序列长度
            
        Returns:
            X: 特征数据
            y: 目标数据
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X, y, train_ratio=0.8):
        """
        分割训练集和测试集
        
        Args:
            X: 特征数据
            y: 目标数据
            train_ratio: 训练集比例
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        split_point = int(len(X) * train_ratio)
        
        X_train = X[:split_point]
        X_test = X[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]
        
        logger.info(f"数据分割: 训练集{len(X_train)}, 测试集{len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_missing_values(self, data, method='forward_fill'):
        """
        处理缺失值
        
        Args:
            data: DataFrame
            method: 处理方法 ('forward_fill', 'backward_fill', 'interpolate')
            
        Returns:
            处理后的DataFrame
        """
        if method == 'forward_fill':
            return data.fillna(method='ffill').fillna(method='bfill')
        elif method == 'backward_fill':
            return data.fillna(method='bfill').fillna(method='ffill')
        elif method == 'interpolate':
            return data.interpolate()
        else:
            raise ValueError(f"未知的处理方法: {method}")
    
    def remove_outliers(self, data, columns=None, std_threshold=3):
        """
        移除异常值（使用3-sigma规则）
        
        Args:
            data: DataFrame
            columns: 要处理的列
            std_threshold: 标准差阈值
            
        Returns:
            处理后的DataFrame
        """
        df = data.copy()
        
        if columns is None:
            columns = df.columns
        
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] > mean - std_threshold*std) & 
                    (df[col] < mean + std_threshold*std)]
        
        logger.info(f"移除异常值后数据形状: {df.shape}")
        return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # 测试
    from data_fetcher import DataFetcher
    
    fetcher = DataFetcher('000960.SZ')
    data = fetcher.fetch_data(start_date='2023-01-01', end_date='2026-06-13')
    
    preprocessor = DataPreprocessor()
    
    # 提取收盘价
    prices = data['Close'].values.reshape(-1, 1)
    
    # 归一化
    normalized_prices = preprocessor.normalize(prices)
    print(f"归一化后数据范围: [{normalized_prices.min():.4f}, {normalized_prices.max():.4f}]")
    
    # 创建序列
    X, y = preprocessor.create_sequences(normalized_prices, seq_length=60)
    print(f"序列数据形状: X={X.shape}, y={y.shape}")
    
    # 分割数据
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
