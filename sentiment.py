from flask import Flask, jsonify
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import requests


app = Flask(__name__)




@app.route('/sentiment', methods=['GET','POST'])
def sentiment():
    api_url = "http://192.168.1.21:9000/user/turf_rating/"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
    
    df = pd.DataFrame(data, columns=['turfid', 'rating', 'review'])
    
    nltk.download('vader_lexicon')
    
    # Create a SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Add a new column 'sentiment' to the DataFrame
    df['sentiment'] = df['review'].apply(lambda x: 'positive' if sid.polarity_scores(x)['compound'] >= 0 else 'negative')

    # df['weighted_rating'] = df.apply(lambda row: row['rating'] * (1.2 if row['sentiment'] == 'positive' else 0.8), axis=1)
    df['weighted_rating'] = df.apply(lambda row: min(row['rating'] * (1.2 if row['sentiment'] == 'positive' else 0.8), 5), axis=1)
    # df['weighted_rating'] = df.apply(lambda row: round(min(row['rating'] * (1.2 if row['sentiment'] == 'positive' else 0.8), 5), 1), axis=1)
    # df['weighted_rating']=round(df['weighted_rating'],1)
 
 
    # Calculate the overall rating for each turf
    overall_rating = round(df.groupby('turfid')['weighted_rating'].mean().reset_index(),1)

    # Convert the result to JSON
    result = overall_rating.to_dict(orient='records')

    return result

if __name__ == '__main__':
    app.run(debug=True)









# from flask import Flask, request, jsonify
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import nltk
# nltk.download('vader_lexicon')

# app = Flask(__name__)

# api_url = "http://192.168.1.18:9000/turf_owner_app/review/"

# @app.route('/analyze_reviews', methods=['POST'])
# def analyze_reviews():
#     try:
#         data = request.json

#         # Assuming your API returns reviews as a list under the key 'reviews'
#         reviews = data.get('reviews', [])

#         # Initialize the sentiment analyzer
#         sid = SentimentIntensityAnalyzer()

#         # Analyze each review and classify as positive or negative based on a threshold (e.g., 4 stars)
#         positive_reviews = []
#         negative_reviews = []

#         for review in reviews:
#             # Extract relevant information from the review
#             turf_id = review.get('turf_id', '')
#             user_id = review.get('user_id', '')
#             user_rating = review.get('rating', 0)
#             review_text = review.get('review', '')

#             # Use NLTK VADER to get the sentiment score
#             sentiment_score = sid.polarity_scores(review_text)

#             # Classify as positive or negative based on the user rating
#             if user_rating >= 4:
#                 sentiment = 'positive'
#                 positive_reviews.append({
#                     'turf_id': turf_id,
#                     'user_id': user_id,
#                     'review': review_text,
#                     'sentiment': sentiment
#                 })
#             else:
#                 sentiment = 'negative'
#                 negative_reviews.append({
#                     'turf_id': turf_id,
#                     'user_id': user_id,
#                     'review': review_text,
#                     'sentiment': sentiment
#                 })

#         result = {
#             'positive_reviews': positive_reviews,
#             'negative_reviews': negative_reviews
#         }

#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
