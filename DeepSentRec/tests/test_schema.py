from utils.schema import map_rating_to_sentiment
import pandas as pd

def test_sentiment_mapping():
    s = pd.Series([1,2,3,4,5,None])
    out = map_rating_to_sentiment(s)
    assert out.iloc[0] == "negative"
    assert out.iloc[2] == "neutral"
    assert out.iloc[4] == "positive"
