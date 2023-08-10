import pickle
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import pandas as pd

train_data_path = "new_train.csv"
val_data_path = "new_val.csv"

train_df = pd.read_csv(train_data_path)
train_df = train_df.to_numpy()

X_train = train_df[:, 0:]
t_train = train_df[:, 0]

val_df = pd.read_csv(val_data_path)
val_df = val_df.to_numpy()

X_val = val_df[:, 0:]
t_val = val_df[:, 0]

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, t_train)
pred = ridge.predict(X_val)
print(r2_score(t_val, pred))

filename = 'finalized_model.pkl'
pickle.dump(ridge, open(filename, 'wb'))