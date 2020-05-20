import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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

#add cross validate,
from sklearn.model_selection import KFold

# 3-fold
cv = KFold(n_splits = 3, random_state = 1, shuffle = True)

max_score = 0
max_n_tree = 0
max_m_depth = 0

#for n_tree in range(36, 121, 12):
#    local_max_score = 0
#    local_max_m_depth = 0
#    for m_depth in range(15, 26, 5):
#        print(n_tree, m_depth)
#        rfc = RandomForestClassifier(n_estimators=n_tree, max_depth=m_depth, n_jobs = -1)
#        scores = []
#        for train_index, test_index in cv.split(X):
#            #print(train_index)
#            X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index]
#            rfc.fit(X_train, y_train)
#            scores.append(rfc.score(X_test, y_test))
#        average_score = np.mean(scores)
#        if average_score > local_max_score:
#            local_max_score = average_score
#            local_max_m_depth = m_depth
#    print("***")
#    print(n_tree, local_max_m_depth, local_max_score)
#    print("***")
#    if local_max_score > max_score:
#        max_score, max_n_tree, max_m_depth = local_max_score, n_tree, local_max_m_depth

#print(max_n_tree, max_m_depth)

max_n_tree = 55
max_m_depth = 15

model = RandomForestClassifier(n_estimators = max_n_tree, max_depth = max_m_depth, n_jobs = -1, verbose = 2)
model.fit(X, y)

print("Finished training!")

# X_test = pd.get_dummies(test_data, columns = features[1:])
X_test = test_data[features[1:]]
predictions = model.predict_proba(X_test)

#print(predictions[:,1])

result = pd.DataFrame({'value': predictions[:,1]})
result.to_csv("result.csv", index = False)
