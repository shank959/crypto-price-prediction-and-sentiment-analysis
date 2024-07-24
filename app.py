from flask import Flask, request, jsonify, render_template
from sentiment_analysis import sentiment_score
from webscrape import scrape_news

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    articles = request.json.get('articles', [])
    sentiments = [sentiment_score(article) for article in articles]
    return jsonify(sentiments)


if __name__ == '__main__':
    app.run(debug=True)
