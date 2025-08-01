import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import os


def load_raw_data(path):
    df=pd.read_csv(path)
    df['Date']= pd.to_datetime(df['Date'],dayfirst=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def clean_data(df):
    df=df[df['Volume']>0].copy() #Remove non trading days
    df.dropna(subset=['Open', 'High', 'Low','Close','Volume'], inplace=True) #dropn rows with missing OHLC values
    if 'Adj close' in df.columns:
        df['Price']=df['Adj Close']
    else:
        df['Price']=df['Close']
    
    #Add technical indicators
    df['SMA_10']=ta.sma(df['Close'],length=10)
    df['RSI_14']=ta.rsi(df['Close'], length=14)
    
    df.dropna(inplace=True)
    
    scaler=MinMaxScaler()
    df[['Price','SMA_10','RSI_14']]=scaler.fit_transform(df[['Price','SMA_10','RSI_14']])
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path)
    print(f"Cleaned data saved to : {output_path}")
    
if __name__=="__main__":
    raw_path="/Users/agrimjoshi/Desktop/CMPT 310 Project/data/raw/stock_market_data/nasdaq/csv/AAPL.csv"
    cleaned_path="/Users/agrimjoshi/Desktop/CMPT 310 Project/data/Processed/AAPL_cleaned.csv"
    
    df=load_raw_data(raw_path)
    df_cleaned=clean_data(df)
    save_cleaned_data(df_cleaned, cleaned_path)
        