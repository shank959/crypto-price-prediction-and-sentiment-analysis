import streamlit as st
from utils.sentiment_analysis import sentiment_scores, sentiment_count
from utils.webscrape import scrape_news
from utils.helper_functions import create_pie_chart, create_table_df

# List of coin options
coin_options = [
    "Bitcoin",
    "Ethereum",
    "Solana",
    "Dogecoin",
    "Cardano",
    "Tether",
    "Tron",
    "Polkadot",
    "Chainlink",
    "Polygon",
    "Litecoin"
]

# Main function for the Streamlit app
def main():
    st.title("Cryptocurrency Sentiment Analyzer")
    
    # Sidebar or Main Page: Coin selection
    st.subheader("Select a cryptocurrency:")
    coin = st.selectbox("Cryptocurrency", coin_options)
    
    # Handle invalid selection or other feedback
    st.write("Choose a cryptocurrency to analyze sentiment from recent news articles.")
    
    # Form submission to analyze sentiment
    if st.button("Analyze"):
        articles = scrape_news(coin_name=coin, num_pages=1)
        
        if not articles:
            st.error(f"No articles found for {coin}. Please try again.")
        else:
            # Perform sentiment analysis
            sentiments = sentiment_scores(articles)
            counts = sentiment_count(sentiments)
            df = create_table_df(sentiments=sentiments)
            
            # Display results
            st.write("")
            st.write(f"## Sentiment Analysis Results for {coin}:")
            
            # Show sentiment counts
            st.write("#### Sentiment Counts")
    
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positve", counts[0])
            with col2:
                st.metric("Neutral", counts[1])
            with col3:
                st.metric("Negative", counts[2])



            
            # Display list of articles
            st.dataframe(df)  

            # Show pie chart
            st.write("### Sentiment Distribution")
            fig = create_pie_chart(counts)
            st.plotly_chart(fig)  # Render the pie chart

if __name__ == "__main__":
    main()
