import pandas as pd
import numpy as np
import time

def avg_amount_loans_prev(df):
    avg = pd.Series(index=df.index, dtype=np.float64)
    for i in df.index:
        df_aux = df.loc[df.loan_date < df.loan_date.loc[i], :]
        avg.at[i] = df_aux.loan_amount.mean()
    return avg

start_time = time.time()

df = pd.read_csv('dataset_credit_risk.csv')

df = df.sort_values(by=["id", "loan_date"])
df = df.reset_index(drop=True)
df["loan_date"] = pd.to_datetime(df.loan_date)


df_grouped = df.groupby("id")
df["nb_previous_loans"] = df_grouped["loan_date"].rank(method="first") - 1

avg_amount_loans_previous = pd.Series(dtype=np.object)
# the following cycle is the one that takes forever if we try to compute it for the whole dataset
for user in df.id.unique():
    df_user = df.loc[df.id == user, :]
    avg_amount_loans_previous = avg_amount_loans_previous.append(avg_amount_loans_prev(df_user))

df["avg_amount_loans_previous"] = avg_amount_loans_previous

df['birthday'] = pd.to_datetime(df['birthday'], errors='coerce')

df['age'] = (pd.to_datetime('today').normalize() - df['birthday']).dt.days // 365

df['job_start_date'] = pd.to_datetime(df['job_start_date'], errors='coerce')

df['years_on_the_job'] = (pd.to_datetime('today').normalize() - df['job_start_date']).dt.days // 365

df['flag_own_car'] = df.flag_own_car.apply(lambda x : 0 if x == 'N' else 1)

df = df[['id', 'age', 'years_on_the_job', 'nb_previous_loans', 'avg_amount_loans_previous', 'flag_own_car', 'status']]

df.to_csv('train_model.csv', index=False)
    
end_time = time.time()

print("Total Time:", end_time - start_time, "seconds.")
