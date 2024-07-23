from sentiment_analysis import sentiment_score
from webscrape import scrape_news, retrieve_articles
import matplotlib.pyplot as plt


def main():

    hrefs = scrape_news(coin_name='cardano', num_pages=3)
    if not hrefs:
        print("Error: No News Articles Found")
        exit()
    else:
        print("Article hrefs retrieved successfully...")

    # for title, href in hrefs.items():     #! display hrefs
    #     print(f"{title}: {href}")

    articles = retrieve_articles(hrefs)
    if articles:
        print("Article texts retrieved successfully...")

    # for key, value in articles.items():   #! display articles
    #     print(f"{key}: {value}")
    #     break

    sentiment_table = {}
    for key, value in articles.items():
        text = f"{key} {value}"
        sentiment = sentiment_score(text)
        label = 'Positive' if sentiment == 1 else 'Negative' if sentiment == -1 else 'Neutral'
        # print(f"{key}: {label} ({confidence.round(3)})")
        sentiment_table[key] = label

    if sentiment_table:
        print("Sentiment analysis performed successfully...")


    # VISUALIZING THE DATA
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for sentiment in sentiment_table.values():
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1

    for sentiment in sentiment_counts:
        print(f"{sentiment}: {sentiment_counts[sentiment]}")

    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    colors = ['#66b3ff', '#ff6666', '#ffcc99']
    explode = (0.1, 0, 0)  # explode the 1st slice (Positive)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Pie chart
    ax[0].pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    ax[0].axis('equal')
    ax[0].set_title('Sentiment Distribution (Pie Chart)')

    # Bar chart
    ax[1].bar(labels, sizes, color=colors)
    ax[1].set_title('Sentiment Distribution (Bar Chart)')
    ax[1].set_xlabel('Sentiment')
    ax[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
