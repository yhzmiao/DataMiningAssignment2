import pandas as pd
import numpy as np
from imblearn.ensemble import EasyEnsembleClassifier

train_data = pd.read_csv("output_train.csv")
test_data = pd.read_csv("output_test.csv")

features = [
    'booking_bool',
    'site_id',
    'month',
    'year',
    #'visitor_location_country_id'
    'prop_starrating',
    'prop_brand_bool',
    'location_score1',
    'log_historical_price',
    'price_group',
    'promotion_flag',
    #'srch_destination_id',
    'length_of_stay',
    'booking_window',
    'srch_adults_count',
    'srch_children_count',
    'srch_room_count',
    'srch_saturday_night_bool',
    'random_bool',
    'foreigner',
    'preferred',
    'strong_competitor',
    'query_aff',
    'distance',
    'location_score2',
    'prop_review_score',
    'family_type'
]

train_data = train_data[features]
#train_data.info()
test_data = test_data[features[1:]]
#test_data.info()

print("Finished data preprocessing!")

#X = pd.get_dummies(train_data[features[1:]], columns = features[1:])
X = train_data[features[1:]]
y = train_data['booking_bool']

# add cross validate,
from sklearn.model_selection import KFold

# 5-fold
cv = KFold(n_splits = 3, random_state = 1, shuffle = True)

max_score = 0
max_n_estimator = 0

#for n_estimator in range(30, 100, 10):
#    print(n_estimator)
#    abc = AdaBoostClassifier(n_estimators=n_estimator, random_state = 0)
#    scores = []
#    for train_index, test_index in cv.split(X):
#        X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index]
#        abc.fit(X_train, y_train)
#        scores.append(abc.score(X_test, y_test))
#    average_score = np.mean(scores)
#    if average_score > max_score:
#        max_score, max_n_estimator = average_score, n_estimator
#    print(n_estimator, average_score)

max_n_estimator = 15
print(max_n_estimator)

model = EasyEnsembleClassifier(n_estimators = max_n_estimator, random_state = 0)
model.fit(X, y)

print("Finished training!")

#X_test = pd.get_dummies(test_data, columns = features[1:])
X_test = test_data[features[1:]]

predictions = model.predict_proba(X_test)

result = pd.DataFrame({'value': predictions[:,0]})
result.to_csv("result.csv", index = False)
