import plotly.express as px
import plotly.io as pio
import pandas as pd

def create_table_df(sentiments):

    articles = list(sentiments.items())
    df = pd.DataFrame(articles, columns=['Article Title', 'Sentiment'])

    def categorize_sentiment(score):
        if score > 0:
            return 'Positive'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Negative'
    
    df['Sentiment'] = df['Sentiment'].apply(categorize_sentiment)

    def sentiment_color(val):
        color = ''
        if val == 'Positive':
            color = 'green'
        elif val == 'Neutral':
            color = 'gray'
        else:
            color = 'red'
        return f'color: {color}'

    df = df.style.applymap(sentiment_color, subset=['Sentiment'])
    return df


def create_pie_chart(counts):

    data = pd.DataFrame({
        'Sentiment': ['Positive', 'Neutral', 'Negative'],
        'Counts': counts
    })

    # Create pie chart using Plotly
    fig = px.pie(data, names='Sentiment', values='Counts', title="Sentiment Distribution", 
                 color='Sentiment', color_discrete_map={
                    'Positive': 'green',
                    'Neutral': 'grey',
                    'Negative': 'red'
                 })
    
    # Update layout to change the title and legend text colors
    fig.update_layout(
        paper_bgcolor='rgb(240, 242, 245)',  
        plot_bgcolor='rgb(240, 242, 245)',
        # title={
        #     'text': "Sentiment Distribution",
        #     'y': 0.99,
        #     'x': 0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top',
        #     'font': {
        #         'color': 'black',   
        #         'size': 24,        
        #         'family': 'Arial'  
        #     }
        # },
        legend=dict(
            title_font=dict(
                color='red'         
            ),
            font=dict(
                color='black',     
                size=12
            ),
            orientation="h",  
            yanchor="bottom",
            y=0.05,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig
