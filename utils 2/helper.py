import plotly.express as px
import plotly.io as pio
import pandas as pd


def create_pie_chart(counts):

    data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Counts': counts
        })

    fig = px.pie(data, names='Sentiment', values='Counts', title="Sentiment Distribution", color='Sentiment', color_discrete_map={
                    'Positive': 'green',
                    'Neutral': 'grey',
                    'Negative': 'red'
                })
    
    fig.update_layout(
            paper_bgcolor='rgb(240, 242, 245)',
            title={
                'text': "Sentiment Distribution",
                'y':0.99,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.05,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            
        )
    
    return pio.to_html(fig, full_html=False)