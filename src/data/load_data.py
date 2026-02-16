import pandas as pd
import os
file_path ="/Users/geetheswarreddy/Desktop/risk_1/data/raw_data/raw.csv"
def load_data(file_path):
    try:
        data=pd.read_csv(file_path)
        data.drop(columns="Student_ID", inplace=True)
        data.rename(columns={"Depression":"risk_flag"}, inplace=True)
        data["risk_flag"]=data["risk_flag"].astype(int)
        print(data.head())
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__=="__main__":
    data=load_data(file_path)


