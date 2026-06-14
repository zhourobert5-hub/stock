# ===== 锡业股份预测系统配置文件 =====

import os
from datetime import datetime

# ===== 基本配置 =====
PROJECT_NAME = "锡业股份预测系统"
VERSION = "1.0.0"
AUTHOR = "Stock Prediction Team"
CREATED_DATE = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ===== 股票配置 =====
STOCK_TICKER = "000960.SZ"  # 锡业股份代码
STOCK_NAME = "锡业股份"
STOCK_START_DATE = "2022-01-01"
STOCK_END_DATE = None  # 使用今日

# ===== 数据配置 =====
DATA_DIR = "data"
MODELS_DIR = "models"
BACKTEST_DIR = "backtest/backtest_results"
LOGS_DIR = "logs"

# 创建目录
for directory in [DATA_DIR, MODELS_DIR, BACKTEST_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ===== LSTM模型参数 =====
LSTM_CONFIG = {
    'seq_length': 60,              # 序列长度（天数）
    'feature_scale': (0, 1),       # 数据标准化范围
    'train_ratio': 0.8,            # 训练集比例
    'test_ratio': 0.2,             # 测试集比例
    'batch_size': 32,              # 批次大小
    'epochs': 100,                 # 训练轮数
    'validation_split': 0.2,       # 验证集比例
    'lstm_units': [50, 50, 50],    # LSTM层单元数
    'dropout_rate': 0.2,           # Dropout率
    'optimizer': 'adam',           # 优化器
    'loss': 'mse',                 # 损失函数
}

# ===== Prophet模型参数 =====
PROPHET_CONFIG = {
    'interval_width': 0.95,        # 置信区间
    'yearly_seasonality': True,    # 年度季节性
    'weekly_seasonality': True,    # 周度季节性
    'daily_seasonality': False,    # 日度季节性
    'seasonality_mode': 'additive', # 季节性模式
}

# ===== 融合模型配置 =====
ENSEMBLE_CONFIG = {
    'lstm_weight': 0.7,            # LSTM权重
    'prophet_weight': 0.3,         # Prophet权重
    'voting_method': 'weighted_average',  # 投票方法
}

# ===== 回测配置 =====
BACKTEST_CONFIG = {
    'initial_capital': 100000,     # 初始资本（元）
    'position_size': 0.1,          # 单笔仓位（初始资本的比例）
    'stop_loss': 0.05,             # 止损比例（5%）
    'take_profit': 0.15,           # 止盈比例（15%）
    'commission': 0.001,           # 交易佣金（0.1%）
    'slippage': 0.001,             # 滑点（0.1%）
}

# ===== 交易机器人配置 =====
TRADER_BOT_CONFIG = {
    'enabled': True,               # 是否启用
    'trading_hours': {
        'open': '09:30',           # 开盘时间
        'close': '15:00',          # 收盘时间
    },
    'prediction_threshold': 0.55,  # 预测置信度阈值
    'max_positions': 2,            # 最大持仓数
    'check_interval': 300,         # 检查间隔（秒）
}

# ===== 数据源配置 =====
DATA_SOURCE_CONFIG = {
    'primary': 'yfinance',         # 主数据源
    'backup': 'tushare',           # 备用数据源
    'cache_enabled': True,         # 启用缓存
    'cache_ttl': 86400,            # 缓存有效期（秒）
}

# ===== 预测配置 =====
PREDICTION_CONFIG = {
    'forecast_days': 5,            # 预测天数
    'confidence_level': 0.95,      # 置信度
    'min_data_points': 100,        # 最少数据点
    'rebalance_frequency': 'weekly', # 重平衡频率
}

# ===== 日志配置 =====
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(LOGS_DIR, 'app.log'),
}

# ===== API配置 =====
API_CONFIG = {
    'tushare_token': 'YOUR_TUSHARE_TOKEN_HERE',  # TuShare API token
    'timeout': 30,                 # 请求超时（秒）
    'max_retries': 3,              # 最大重试次数
}

# ===== 通知配置 =====
NOTIFICATION_CONFIG = {
    'enabled': False,              # 是否启用通知
    'channels': ['email', 'wechat'],  # 通知渠道
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'sender': 'your_email@gmail.com',
        'password': 'your_password',
    },
}

# ===== 模型评估指标 =====
EVALUATION_METRICS = {
    'rmse': True,                  # 均方根误差
    'mae': True,                   # 平均绝对误差
    'mape': True,                  # 平均绝对百分比误差
    'r2_score': True,              # R²分数
    'directional_accuracy': True,  # 方向准确率
}

# ===== 打印配置信息 =====
if __name__ == '__main__':
    print(f"项目名称: {PROJECT_NAME}")
    print(f"版本: {VERSION}")
    print(f"创建时间: {CREATED_DATE}")
    print(f"股票代码: {STOCK_TICKER}")
    print(f"LSTM序列长度: {LSTM_CONFIG['seq_length']}")
    print(f"预测天数: {PREDICTION_CONFIG['forecast_days']}")
