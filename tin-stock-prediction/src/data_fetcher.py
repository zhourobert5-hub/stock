# ===== 数据获取模块 =====

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class DataFetcher:
    """获取股票历史数据"""
    
    def __init__(self, ticker, cache_dir='data'):
        self.ticker = ticker
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f'{ticker}_cache.csv')
        os.makedirs(cache_dir, exist_ok=True)
        
    def fetch_data(self, start_date=None, end_date=None, use_cache=True):
        """
        获取股票数据
        
        Args:
            start_date: 开始日期 (str or datetime)
            end_date: 结束日期 (str or datetime)
            use_cache: 是否使用缓存
            
        Returns:
            DataFrame: 股票数据
        """
        try:
            # 检查缓存
            if use_cache and os.path.exists(self.cache_file):
                logger.info(f"从缓存加载数据: {self.cache_file}")
                data = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
                return data
            
            # 从yfinance获取数据
            logger.info(f"从yfinance获取数据: {self.ticker}")
            data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"无法获取 {self.ticker} 的数据")
            
            # 保存到缓存
            data.to_csv(self.cache_file)
            logger.info(f"数据已保存到缓存: {self.cache_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            raise
    
    def add_technical_indicators(self, data):
        """
        添加技术指标
        
        Args:
            data: DataFrame
            
        Returns:
            DataFrame: 包含技术指标的数据
        """
        df = data.copy()
        
        # MA均线
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 成交量MA
        df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
        
        # 日涨跌幅
        df['Daily_Return'] = df['Close'].pct_change()
        
        return df.dropna()
    
    def get_latest_price(self):
        """获取最新价格"""
        data = yf.download(self.ticker, period='1d', progress=False)
        return data['Close'].iloc[-1]
    
    def clear_cache(self):
        """清除缓存"""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            logger.info(f"缓存已清除: {self.cache_file}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # 测试
    fetcher = DataFetcher('000960.SZ')
    data = fetcher.fetch_data(start_date='2023-01-01', end_date='2026-06-13')
    print(f"数据形状: {data.shape}")
    print(f"最新收盘价: {fetcher.get_latest_price()}")
    
    # 添加指标
    data_with_indicators = fetcher.add_technical_indicators(data)
    print(data_with_indicators.tail())
