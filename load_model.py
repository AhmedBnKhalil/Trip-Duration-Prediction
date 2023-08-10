import pickle

import pandas as pd

val_data_path = "new_val.csv"
val_df = pd.read_csv(val_data_path)
val_df = val_df.to_numpy()

X_val = val_df[:, 0:]
t_val = val_df[:, 0]
loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))
result = loaded_model.score(X_val, t_val)
print(result)