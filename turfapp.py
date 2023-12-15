from flask import Flask, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

app = Flask(__name__)


model = joblib.load('anomaly_detection_model.joblib')


data=[
    {'turf_id': 1, 'owner_id': 'A001', 'price': 1000, 'monthly_earnings': 100000},
    {'turf_id': 2, 'owner_id': 'A002', 'price': 1200, 'monthly_earnings': 120000},
    {'turf_id': 3, 'owner_id': 'A003', 'price': 800, 'monthly_earnings': 70000},
    {'turf_id': 4, 'owner_id': 'A004', 'price': 1100, 'monthly_earnings': 90000},
    {'turf_id': 5, 'owner_id': 'A005', 'price': 1300, 'monthly_earnings': 110000},
    {'turf_id': 6, 'owner_id': 'A006', 'price': 600, 'monthly_earnings': 95000},
    {'turf_id': 7, 'owner_id': 'A007', 'price': 1000, 'monthly_earnings': 105000},
    {'turf_id': 8, 'owner_id': 'A008', 'price': 1200, 'monthly_earnings': 80000},
    {'turf_id': 9, 'owner_id': 'A009', 'price': 2500, 'monthly_earnings': 120000},
    {'turf_id': 10, 'owner_id': 'A010', 'price': 1100, 'monthly_earnings': 100000},
]




@app.route('/api/send_notifications', methods=['GET'])
def send_notifications():
    df = pd.DataFrame(data)
    df['Outlier'] = model.predict(df[['price']])
    outliers_df = df[df['Outlier'] == -1]
    inliers_df = df[df['Outlier'] == 1]
    
    
    median = outliers_df['price'].median()
    min_value = inliers_df['monthly_earnings'].min()
    print(min_value)
    owner_notification = outliers_df[outliers_df["monthly_earnings"]<min_value]
    print(outliers_df)

    notification_message = "Notification: Your turf's monthly earnings are relatively lower compared to other turfs. Consider implementing specific measures such as promoting events, adjusting pricing, or enhancing the turf facilities to attract more customers and increase earnings. If needed, feel free to discuss strategies with the management team. Thank you for your attention and efforts to improve your turf's performance."
    print(owner_notification)
    owner_id = "aghgxs"
    
    for index, row in owner_notification.iterrows():
        print(row)
        print(index)
        owner_id = row['owner_id']
        # print(owner_id)
    return jsonify({'message': notification_message, 'id': owner_id})   
    

if __name__ == '__main__':
    app.run(debug=True)
