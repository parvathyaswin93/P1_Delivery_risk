import pandas as pd

def load_data():
  df=pd.read_csv(r"D:\internship\project\P1_Delivery_Risk\P1_Delivery_risk\data\Delivery risk predictor.csv")
  print(df)
load_data()
