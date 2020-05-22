import pandas as pd

test_data = pd.read_csv("test_set_VU_DM.csv")
output = pd.read_csv("result.csv")

result = pd.DataFrame({'srch_id': test_data.srch_id, 'prop_id': test_data.prop_id, 'value': output.value})
output = result.sort_values(by=['srch_id', 'value'], ascending = (True, False))[['srch_id', 'prop_id']]

output.to_csv('my_submission_13.csv', index=False)
print('Your submission was successfully saved!')
