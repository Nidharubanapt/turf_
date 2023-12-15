from flask import Flask, jsonify
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import requests


app = Flask(__name__)




@app.route('/sentimentreview', methods=['GET','POST'])
def sentiment():
    api_url = "http://192.168.1.21:9000/user/turf_rating/"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
    
    df = pd.DataFrame(data, columns=['turfid', 'rating', 'review'])
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df['review'].apply(lambda x: 'positive' if sid.polarity_scores(x)['compound'] >= 0 else 'negative')
    df
    
    weight_numerical_rating = 0.8  # Increased the weight for the numerical rating
    weight_sentiments = .15
    weight_number_of_reviews = 0.05
    
    overall_ratings_per_turf = pd.DataFrame(columns=['turfid', 'overall_rating'])
    for turf_id, turf_data in df.groupby('turfid'):
        average_numerical_rating = turf_data['rating'].mean()
        positive_sentiments = (turf_data['sentiment'] == 'positive').sum()  # Sum of positive sentiments
        total_reviews = turf_data['turfid'].nunique()
        max_total_reviews = turf_data['turfid'].value_counts().max()

        overall_rating = (average_numerical_rating * weight_numerical_rating +
                        (positive_sentiments / total_reviews) * weight_sentiments +
                        (total_reviews / max_total_reviews) * weight_number_of_reviews)
        
        scaled_overall_rating = min(overall_rating, 5.0)

        # Concatenate the result to the overall_ratings_per_turf DataFrame
        overall_ratings_per_turf = round(pd.concat([overall_ratings_per_turf,
                                          pd.DataFrame({'turfid': [turf_id], 'overall_rating': [scaled_overall_rating]})],
                                         ignore_index=True),1)
  
    result = overall_ratings_per_turf.to_dict(orient='records')

    return result
       
       
       
       
       
if __name__ == '__main__':
    app.run(debug=True)
