import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

train_data = pd.read_csv("output_train_ori.csv")

features = [
    'booking_bool',
    'site_id',
    'month',
    'year',
    #'visitor_location_country_id'
    'prop_starrating',
    'prop_brand_bool',
    'prop_location_score1',
    'prop_log_historical_price',
    'price_usd',
    'promotion_flag',
    #'srch_destination_id',
    'srch_length_of_stay',
    'srch_booking_window',
    'srch_adults_count',
    'srch_children_count',
    'srch_room_count',
    'srch_saturday_night_bool',
    'random_bool',
    'foreigner',
    'preferred',
    'strong_competitor',
    'srch_query_affinity_score',
    'orig_destination_distance',
    'prop_location_score2',
    'prop_review_score',
    'family_type',
    'cost_effective'
]

train_data = train_data[features]

print("Finished data preparation!")

#X = pd.get_dummies(train_data[features[1:]], columns = features[1:])
X_ori = train_data[features[1:]]
y_ori = train_data['booking_bool']

sm = SMOTE(random_state = 42, n_jobs = -1)
X, y = sm.fit_resample(X_ori, y_ori)

print("Finished Smote!")

result = X
result['booking_bool'] = y
result.to_csv("output_train_smote.csv")
