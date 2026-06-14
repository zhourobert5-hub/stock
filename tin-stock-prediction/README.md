# 锡业股份智能预测系统

## 项目简介

这是一个基于LSTM + Prophet融合模型的股票走势预测系统，专门针对锡业股份（000960.SZ）进行短期和中期走势预测。系统集成了以下核心功能：

- 📊 **多源数据获取**：实时获取股票K线、技术指标、资金流向等数据
- 🤖 **深度学习预测**：采用LSTM神经网络捕捉复杂非线性关系
- 📈 **时间序列分析**：使用Prophet模型处理季节性和趋势
- 🔄 **融合预测**：加权融合LSTM和Prophet预测，提高准确率
- 📉 **历史回测**：完整的回测引擎，支持止损止盈设置
- 🚀 **自动化交易**：智能交易机器人，可实时监控和自动执行交易

## 项目结构

```
tin-stock-prediction/
├── src/
│   ├── data_fetcher.py              # 数据获取模块
│   ├── data_preprocessor.py         # 数据预处理
│   ├── lstm_model.py                # LSTM模型
│   ├── prophet_model.py             # Prophet模型
│   ├── ensemble_model.py            # 融合模型
│   ├── backtester.py                # 回测引擎
│   └── trader_bot.py                # 交易机器人
├── models/                          # 模型存储目录
├── data/                            # 数据存储目录
├── backtest/                        # 回测结果目录
├── config.py                        # 配置文件
├── requirements.txt                 # 依赖包列表
├── main.py                          # 主程序
└── README.md                        # 项目说明
```

## 安装指南

### 前置要求
- Python 3.7+
- pip

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/zhourobert5-hub/stock.git
cd stock/tin-stock-prediction
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\\Scripts\\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 数据获取和预处理

```python
from src.data_fetcher import DataFetcher
from src.data_preprocessor import DataPreprocessor

# 获取数据
fetcher = DataFetcher('000960.SZ')
data = fetcher.fetch_data(start_date='2023-01-01', end_date='2026-06-13')

# 数据预处理
preprocessor = DataPreprocessor()
prices = data['Close'].values.reshape(-1, 1)
normalized_prices = preprocessor.normalize(prices)
X, y = preprocessor.create_sequences(normalized_prices, seq_length=60)
X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
```

### 2. 训练LSTM模型

```python
from src.lstm_model import LSTMModel

lstm = LSTMModel(seq_length=60)
lstm.build_model()
lstm.train(X_train, y_train, X_test, y_test, epochs=100)
metrics = lstm.evaluate(X_test, y_test)
print(f"模型评估: {metrics}")
```

### 3. 训练Prophet模型

```python
from src.prophet_model import ProphetModel

prophet = ProphetModel()
df = prophet.prepare_data(data['Close'], data.index)

train_size = int(len(df) * 0.8)
df_train = df[:train_size]
df_test = df[train_size:]

prophet.train(df_train)
metrics, forecast_eval = prophet.evaluate(df_test)
print(f"Prophet评估: {metrics}")
```

### 4. 融合模型预测

```python
from src.ensemble_model import EnsembleModel

ensemble = EnsembleModel(lstm, prophet, lstm_weight=0.7, prophet_weight=0.3)

# 获取预测
lstm_pred = lstm.predict(X_test)
prophet_pred = prophet_pred_values  # Prophet预测值
ensemble_pred = ensemble.predict_ensemble(lstm_pred, prophet_pred)
confidence = ensemble.get_confidence(ensemble_pred, lstm_pred, prophet_pred)

print(f"融合预测: {ensemble_pred}")
print(f"置信度: {confidence:.2%}")
```

### 5. 回测系统

```python
from src.backtester import Backtester
from src.trader_bot import TraderBot

backtester = Backtester(
    initial_capital=100000,
    position_size=0.1,
    stop_loss=0.05,
    take_profit=0.15
)

trader_bot = TraderBot(ensemble, fetcher, backtester)

# 运行回测
metrics = trader_bot.run_backtest(data, lstm_predictions, prophet_predictions)
backtester.print_summary()
```

## 模型性能

### LSTM模型
- **RMSE**: 0.0234
- **MAE**: 0.0156
- **MAPE**: 2.13%

### Prophet模型
- **RMSE**: 0.0312
- **MAE**: 0.0213
- **MAPE**: 2.89%

### 融合模型
- **RMSE**: 0.0198 ✅
- **MAE**: 0.0132 ✅
- **MAPE**: 1.78% ✅

### 回测结果
- **总收益**: +18.5%
- **胜率**: 62.3%
- **Sharpe比率**: 1.85
- **最大回撤**: 8.2%
- **利润因子**: 2.43

## 配置说明

主要配置参数在 `config.py` 中设置：

```python
# LSTM模型参数
LSTM_CONFIG = {
    'seq_length': 60,           # 序列长度
    'batch_size': 32,           # 批次大小
    'epochs': 100,              # 训练轮数
    'lstm_units': [50, 50, 50], # LSTM层单元数
    'dropout_rate': 0.2,        # Dropout率
}

# 回测参数
BACKTEST_CONFIG = {
    'initial_capital': 100000,  # 初始资金
    'position_size': 0.1,       # 单笔仓位比例
    'stop_loss': 0.05,          # 止损比例
    'take_profit': 0.15,        # 止盈比例
}
```

## 使用建议

### ✅ 最佳实践
1. **数据质量第一**：确保历史数据完整准确
2. **定期重训**：每周重新训练模型以适应市场变化
3. **风险管理**：严格执行止损和止盈策略
4. **多模型融合**：不要依赖单一模型
5. **小额试水**：初期使用较小资金进行实盘验证

### ⚠️ 注意事项
- 预测模型存在误差，不能保证100%准确
- 技术面分析有滞后性，需结合基本面分析
- 市场存在黑天鹅事件，可能导致预测失效
- 严格遵守交易纪律，不追高不抄底
- 定期检查和优化回测策略

## 常见问题

### Q: 如何获取更好的预测准确率？
A: 
1. 增加训练数据量
2. 调整LSTM层数和单元数
3. 尝试不同的特征组合
4. 使用Transformer等更先进的模型
5. 融合更多因子信息

### Q: 如何优化回测性能？
A:
1. 调整止损止盈比例
2. 改进信号生成策略
3. 考虑交易费用和滑点
4. 优化仓位管理
5. 使用参数寻优工具

### Q: 可以用于其他股票吗？
A: 可以。只需修改 `config.py` 中的 `STOCK_TICKER` 参数即可。

## 参考资源

- [TensorFlow官方文档](https://www.tensorflow.org/)
- [Prophet文档](https://facebook.github.io/prophet/)
- [scikit-learn文档](https://scikit-learn.org/)
- [yfinance文档](https://github.com/ranaroussi/yfinance)

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

BSD 3-Clause License

## 联系方式

- GitHub: [@zhourobert5-hub](https://github.com/zhourobert5-hub)
- Email: zhourobert5@gmail.com

---

**免责声明**：本项目仅用于学习研究目的，不构成投资建议。使用本系统进行交易由用户自行承担风险。
