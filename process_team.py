
import pandas as pd
import numpy as np
from tqdm import tqdm

def date_features_team(df):   
    df['date'] = pd.to_datetime(df['GAME_DATE_EST'], format="%d/%m/%Y")
    df['month']=df['date'].dt.month
    df['dayofweek']=df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['season'] = df['month'] % 12 // 3 + 1
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['weekend'] = df['dayofweek'].map({0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1})
    return df

def lag_features_team(df, n_period,cols): 
    for lag in tqdm(range(1, n_period + 1)):
        for col in tqdm(cols):      
           
            df[f'match_lag_{col}_{lag}'] = df.groupby(['HOME_TEAM_ID','VISITOR_TEAM_ID'])[col].shift(lag)
            
            if 'home' in col:
                groupby_col = 'HOME_TEAM_ID'
            elif 'away' in col:
                groupby_col = 'VISITOR_TEAM_ID'
            df[f'lag_{col}_{lag}'] = df.groupby([groupby_col])[col].shift(lag)
    return df

def rolling_mean_features_team(df,cols,window_size):
    for col in tqdm(cols):
        df[f'match_rolling_mean_{col}_{window_size}'] = df.groupby(['HOME_TEAM_ID','VISITOR_TEAM_ID'])[col].rolling(window=window_size, min_periods=2).mean().reset_index(level=[0, 1], drop=True)
        df[f'match_rolling_mean_{col}_{window_size}'] = df.groupby(['HOME_TEAM_ID','VISITOR_TEAM_ID'])[f'match_rolling_mean_{col}_{window_size}'].shift(fill_value=None)
        if 'home' in col:
            groupby_col = 'HOME_TEAM_ID'
        elif 'away' in col:
            groupby_col = 'VISITOR_TEAM_ID'
        df[f'rolling_mean_{col}_{window_size}'] = df.groupby([groupby_col])[col].rolling(window=window_size, min_periods=2).mean().reset_index(level=0, drop=True)
        df[f'rolling_mean_{col}_{window_size}'] = df.groupby([groupby_col])[f'rolling_mean_{col}_{window_size}'].shift(fill_value=None)
    return df


def preprocess_data_team(df, dict_all):
    cat_cols=['HOME_TEAM','VISITOR_TEAM']
    df[cat_cols] = df[cat_cols].astype(object)
    for col in df[cat_cols]:
        df[col].replace(dict_all[col], inplace=True)
    return df