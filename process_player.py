import pandas as pd

def date_features(df):   
    df['date'] = pd.to_datetime(df['GAME DATE'])
    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month
    df['dayofweek']=df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['season'] = df['month'] % 12 // 3 + 1
    df['weekend'] = df['dayofweek'].map({0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1})
    return df

def lag_features(df, n_period,cols): 
    columns_to_convert = ['FG%', '3P%', 'FT%']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    df[columns_to_convert]=df[columns_to_convert].fillna(0)
    for lag in range(1, n_period + 1):
        for col in cols:      
           
            df[f'match_lag_{col}_{lag}'] = df.groupby(['PLAYER','TEAM'])[col].shift(lag)
            
            groupby_col =['PLAYER','TEAM','MATCH UP']
            df[f'lag_{col}_{lag}'] = df.groupby(groupby_col)[col].shift(lag)
    return df

def rolling_mean_features(df,cols,window_size):
    columns_to_convert = ['FG%', '3P%', 'FT%']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    df[columns_to_convert]=df[columns_to_convert].fillna(0)
    for col in cols:
        df[f'match_rolling_mean_{col}_{window_size}'] = df.groupby(['PLAYER','TEAM'])[col].rolling(window=window_size, min_periods=2).mean().reset_index(level=[0, 1], drop=True)
        df[f'match_rolling_mean_{col}_{window_size}'] = df.groupby(['PLAYER','TEAM'])[f'match_rolling_mean_{col}_{window_size}'].shift(fill_value=None)
    return df

# Function to preprocess data for prediction
def preprocess_data(df, dict_all):
    cat_cols = ['PLAYER', 'TEAM', 'MATCH UP']
    df[cat_cols] = df[cat_cols].astype(object)
    for col in df[cat_cols]:
        df[col].replace(dict_all[col], inplace=True)
    return df
