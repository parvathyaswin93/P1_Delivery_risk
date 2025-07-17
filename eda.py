import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r"D:\internship\project\P1_Delivery_Risk\P1_Delivery_risk\data\Delivery risk predictor.csv")
print(df)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['predicted_risk'].value_counts())
sns.countplot(x='predicted_risk',data=df)
plt.title('prediction_risk')
plt.show()
df['actual_duration_days'].plot.hist()
features = ['planned_duration_days', 'actual_duration_days', 'team_size', 
            'num_bugs', 'num_change_requests', 'delivery_delay_days', 'budget_overrun_pct']
df[features].hist(figsize=(15, 10), bins=30, color='darkgreen')
plt.tight_layout()
plt.show()

sns.heatmap(df[features].corr(), annot=True, cmap="viridis",fmt=".2f")
plt.title("Feature Correlation Heatmap")
#plt.show()

for i in features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='predicted_risk', y=i, data=df, palette='Set1')
    plt.title(f'{i} vs Predicted Risk')
    plt.show()

print(df.groupby('predicted_risk')[features].mean())

df=pd.DataFrame(df)
df["start_date"] = pd.to_datetime(df["start_date"])

df["end_date"] = pd.to_datetime(df["end_date"])
df
df.info()
df['actual_duration'] = (df['end_date'] - df['start_date']).dt.days
df['actual_duration'] 
print((df['actual_duration'] - df['actual_duration_days']).abs().sum())
