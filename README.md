# Cryptocurrency Market Sentiment Analysis Tool

## Overview
This project is designed to analyze the current sentiment of the cryptocurrency market by leveraging a fine-tuned RoBERTa model. It scrapes news articles from various sources, performs sentiment analysis, and visualizes the results.

## Features
- **Sentiment Analysis**: Uses a fine-tuned RoBERTa model to achieve 98% accuracy in classifying the sentiment of financial articles.
- **Web Scraping**: Utilizes Selenium with ChromeDriver and BeautifulSoup to scrape and extract news articles from multiple sources.
- **Data Visualization**: Employs Matplotlib and Seaborn to create insightful visualizations of sentiment trends.

## Requirements
- Python 3.8+
- ChromeDriver (compatible with your version of Chrome)
- Install the necessary Python packages listed in `requirements.txt`.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/crypto-sentiment-analysis.git
    cd crypto-sentiment-analysis
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download and set up [ChromeDriver](https://googlechromelabs.github.io/chrome-for-testing/) and ensure it's in the projects directory.

## Usage

1. **Start the Flask web application on your local device**
    ```bash
    python3 -m app
    ```
2. **Access the local web address to access the home page of the application and choose the crytocurrency to be analyzed**


<img width="1505" alt="Screenshot 2024-07-26 at 4 06 09 PM" src="https://github.com/user-attachments/assets/ea4e6c1c-639b-4be8-a2c6-85ae768afa54">
