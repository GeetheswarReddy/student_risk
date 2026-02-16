import pandas as pd
import os
file_path ="/Users/geetheswarreddy/Desktop/risk_1/data/raw_data/raw.csv"
def load_data(file_path):
    try:
        data=pd.read_csv(file_path)
        print("Data loaded successfully")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    data = load_data(file_path)
    data.drop(columns="Student_ID", inplace=True)
    print(data.head())
if __name__ == "__main__":
    main()