# Cryptocurrency Price Prediction & Market Sentiment Analysis

## Overview
This project is a comprehensive cryptocurrency prediction and sentiment analysis platform that combines **machine learning price prediction** with advanced **news sentiment analysis**. The core system leverages an ensemble XGBoost model trained on historical prices, options data, market indicators, and on-chain data, alongside a highly accurate RoBERTa-based sentiment engine for financial news. Users can forecast 24-hour returns for hundreds of cryptocurrencies and monitor sentiment trends impacting market movements.

## Features

- **Price Prediction Model**: 
  - Ensemble XGBoost trained using historical prices, options data, market indicators, and on-chain metrics to predict 24-hour returns through temporal modeling[1].
  - Demonstrated robust relationship between predicted and actual returns using Spearman rank correlation across 355+ coins, helping identify top gainers and losers[1][2].

- **Sentiment Analysis**:
  - RoBERTa NLP model fine-tuned on a financial article databank (>92% sentiment classification accuracy on validation)[1].
  - Web scraping pipeline (Selenium & BeautifulSoup) to extract fresh news articles for each cryptocurrency ticker[1][3].
  - Sentiment scores are visualized over time for each asset using interactive Plotly dashboards[3].

- **Integrated Forecasting**:
  - Asset-specific sentiment time series and sentiment-drivenness coefficients are fed into the price forecasting algorithm for improved accuracy[1].

- **Data & Evaluation**:
  - Dataset: ~1.37GB of OHLCV parquet files spanning 355+ cryptocurrencies[2].
  - Technical indicators, ETF/options/sentiment/on-chain/fear index integrations.
  - Model evaluation: Weighted Spearman correlation prioritizes rank order performance across many assets[2].

## Requirements

- Python 3.8+
- ChromeDriver (compatible with your Chrome version)
- Install dependencies from `requirements.txt`
- ~1.37GB disk space for historical price and volume data

## Installation

```bash
git clone https://github.com/yourusername/crypto-sentiment-analysis.git
cd crypto-sentiment-analysis
pip install -r requirements.txt
```
Download and set up [ChromeDriver](https://googlechromelabs.github.io/chrome-for-testing/) ensuring it is in your project directory.

## Usage

1. **Launch the Streamlit Application**
    ```bash
    streamlit run app.py
    ```
2. **Select Cryptocurrency & View Dashboard**
   - Access the web app locally
   - Choose your asset and view price forecasts and sentiment trends in real time

## Progress Log

- **Modeling**: 
    - Built a market regime ensemble approach
    - Integrated sentiment, options, ETF, and on-chain data streams for signal diversification

- **Data Handling**: 
    - Created large-scale correlation matrix
    - Automated scraping and data updating

## Example Analysis

A dashboard appears with interactive charts for sentiment distribution and historical price prediction.


***

This platform is built for scalable, automated, and robust crypto asset forecasting, integrating technical and sentiment-driven features for competitive edge in algorithmic trading and market research[1][2][3].

Sources
[1] file.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/67797073/aad45911-dc6f-4cba-998e-2beef72d57d6/file.pdf
[2] README-copy.md https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/67797073/50005ad7-6a09-47e8-8cde-3d8baf42422e/README-copy.md
[3] README.md https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/67797073/4d86733e-3e6f-4e3d-806b-250e52dfbfe5/README.md
