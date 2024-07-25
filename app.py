from flask import Flask, request, render_template, redirect, url_for
from sentiment_analysis import sentiment_scores, sentiment_count
from webscrape import scrape_news

app = Flask(__name__)

@app.route('/')
def index():
    print("index page loaded")
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    coin = request.form['coin']
    articles = scrape_news(coin_name=coin, num_pages=1)
    sentiments = sentiment_scores(articles)
    positive, neutral, negative = sentiment_count(sentiments)
    return render_template('results.html', coin=coin, positive=positive, negative=negative, neutral=neutral)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
