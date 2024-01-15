import streamlit as st
import pandas as pd
import joblib
import pickle
from sklearn.metrics import accuracy_score
from process_team import *
from process_player import *
import warnings
warnings.filterwarnings('ignore')
# Load the trained model
clf = joblib.load('result/model_team.joblib')

# Load the trained model
clf_home = joblib.load('result/model_home_point.joblib')

# Load the trained model
clf_away = joblib.load('result/model_away_point.joblib')


# Load the features
with open('result/features_list_team.pkl', 'rb') as file:
    features_team = pickle.load(file)

# Load the trained model for the second app
clf_player = joblib.load('result/model.joblib')

# Load the features for the second app
with open('result/features_list.pkl', 'rb') as file:
    features_player = pickle.load(file)

# Load the dictionary for categorical encoding
file = open("result/dict_all.obj", 'rb')
dict_all = pickle.load(file)

file = open("result/dict_all_team.obj", 'rb')
dict_all_team = pickle.load(file)


def make_predictions_team(model,model_home,model_away, df, test):
    data=test.copy()
    test['is_test'] = 'yes'
    df = pd.concat([df, test])
    df=df.reset_index(drop=True)
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
    df=df.sort_values(by=['GAME_DATE_EST'])

    feat_cols=['FG_PCT_home','FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
           'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away','REB_away','PTS_home','PTS_away']
    

    df = preprocess_data_team(df, dict_all_team)

    df=lag_features_team(df,5,cols=feat_cols)
    df = rolling_mean_features_team(df, cols=feat_cols, window_size=2)
    df = rolling_mean_features_team(df, cols=feat_cols, window_size=3)
    df = rolling_mean_features_team(df, cols=feat_cols, window_size=4)
    df = rolling_mean_features_team(df, cols=feat_cols, window_size=5)
    df=date_features_team(df)

    test=df.loc[df['is_test']=='yes']
    test = test[features_team]
    data['pred'] = model.predict_proba(test)[:, 1]
    pred_df = data[['HOME_TEAM','VISITOR_TEAM','pred']]

    test_pred_proba = model_home.predict(test)
    bin_labels=['70-80', '81-90', '91-100', '101-110', '111-120', '121-130', '131-200']
    proba_df = pd.DataFrame(test_pred_proba, columns=bin_labels)
    proba_df.insert(loc=0, column='HOME_TEAM', value=data['HOME_TEAM'])
    proba_df = proba_df.applymap(lambda x: f'{float(x):.1%}' if pd.to_numeric(x, errors='coerce') == x else x)



    test_pred_proba_away = model_away.predict(test)
    bin_labels=['70-80', '81-90', '91-100', '101-110', '111-120', '121-130', '131-200']
    proba_df_away = pd.DataFrame(test_pred_proba_away, columns=bin_labels)
    proba_df_away.insert(loc=0, column='VISITOR_TEAM', value=data['VISITOR_TEAM'])
    proba_df_away = proba_df_away.applymap(lambda x: f'{float(x):.1%}' if pd.to_numeric(x, errors='coerce') == x else x)
   
    return pred_df, proba_df,proba_df_away

def make_predictions_player(model, df, test):

    data=test.copy()
    test['is_test'] = 'yes'
    df = pd.concat([df, test])
    df=df.reset_index(drop=True)
    df['GAME DATE'] = pd.to_datetime(df['GAME DATE'], format='mixed', dayfirst=True)

    feat_cols=['FGM', 'FGA', '3PM','FG%', '3P%', 'FT%','MIN','3PA', 'FTM', 'FTA','OREB',
               'DREB','REB','AST','STL','BLK','TOV','+/-','PF','FP']

    df = preprocess_data(df, dict_all)
    df=df.sort_values(by=['GAME DATE'])
    df=lag_features(df,10,cols=feat_cols)
    df = rolling_mean_features(df, cols=feat_cols, window_size=2)
    df = rolling_mean_features(df, cols=feat_cols, window_size=3)
    df = rolling_mean_features(df, cols=feat_cols, window_size=4)
    df = rolling_mean_features(df, cols=feat_cols, window_size=5)
    df=date_features(df)
    
    test=df.loc[df['is_test']=='yes']
    test = test[features_player]
    pred = model.predict_proba(test)
    pred_df = pd.DataFrame(pred, columns=['10-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-100',])
    pred_df=pred_df[['10-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-100']]

    pred_df = pred_df.applymap(lambda x: f'{x:.1%}')
    # pred_df['Max_Label'] = pred_df.idxmax(axis=1)  
    data_columns=['10-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-100']
    pred_df[data_columns] = pred_df[data_columns].apply(lambda x: pd.to_numeric(x.str.rstrip('%'), errors='coerce') / 100)
    # Find the column with the maximum value for each row
    pred_df['predict'] = pred_df[data_columns].idxmax(axis=1)

    pred_df.insert(loc=0, column='PLAYER', value=data['PLAYER'])
    pred_df.insert(loc=1, column='TEAM', value=data['TEAM'])
    pred_df.insert(loc=2, column='MATCH UP', value=data['MATCH UP'])
    pred_df.insert(loc=3, column='GAME DATE', value=data['GAME DATE'])
    pred_df = pred_df.applymap(lambda x: f'{float(x):.1%}' if pd.to_numeric(x, errors='coerce') == x else x)

    if 'PTS' in data.columns:
        pred_df.insert(loc=3, column='PTS', value=data['PTS'])
        bins = [10, 15, 20, 25, 30, 35, 40, 100]
        bin_labels = ['10-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-100']
        pred_df['PTS_bin'] = pd.cut(pred_df['PTS'], bins=bins, labels=bin_labels, right=False)
    return pred_df


def main():
    st.set_page_config(layout="wide")
    st.sidebar.title('NBA Predictions App')
    progress_bar = st.sidebar.header('⚙️ Working Progress')
    progress_bar = st.sidebar.progress(0)
    progress_bar.progress(0)

    # Dropdown selector for prediction options
    prediction_option = st.sidebar.selectbox("Choose Prediction Option", ["Team Prediction", "Player Prediction"])

    # Upload Main Data
    uploaded_file_df = st.sidebar.file_uploader("1.Main датаг оруулна уу (CSV file)", type=["csv"])

    # Upload Test Data
    uploaded_file_test = st.sidebar.file_uploader("2.Test датаг оруулна уу (CSV file)", type=["csv"])

    if uploaded_file_df is not None and uploaded_file_test is not None:
      
        df = pd.read_csv(uploaded_file_df)
        test = pd.read_csv(uploaded_file_test)
        test2=test.copy()
        st.warning("Test Data:")
        st.write(test.head(2))
        progress_bar.progress(40)
  
        # Button to make predictions based on the selected option
        if st.sidebar.button("Predict"):
            if prediction_option == "Team Prediction":
                predictions, predictions2, predictions3 = make_predictions_team(clf, clf_home, clf_away, df, test)
        
                st.success("Result:")
                st.markdown(
                    f'<style>.css-1jw8d53{{max-width: none !important; width: 100vw;}}</style>',
                    unsafe_allow_html=True
                            )   
                st.dataframe(predictions)
                csv_data = predictions.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Татах team",
                    data=csv_data,
                    file_name="prediction_team.csv",
                    key="download_predictions",
                )  

                st.dataframe(predictions2) 
                csv_data1 = predictions2.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Татах home",
                    data=csv_data1,
                    file_name="prediction_home_team.csv",
                    key="download_predictions1",
                )  
                st.dataframe(predictions3) 
                csv_data2 = predictions3.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Татах visiter",
                    data=csv_data2,
                    file_name="prediction_visiter_team.csv",
                    key="download_predictions2",
                )  

            elif prediction_option == "Player Prediction":
                predictions = make_predictions_player(clf_player, df, test)  # Assuming clf_player is the player model
                st.success("Result:")
                st.markdown(
            f'<style>.css-1jw8d53{{max-width: none !important; width: 100vw;}}</style>',
            unsafe_allow_html=True
        )
                st.dataframe(predictions)
                csv_data = predictions.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Татах",
                    data=csv_data,
                    file_name="predictions.csv",
                    key="download_predictions",
                )  

            progress_bar.progress(100)

        # Button to make predictions
        if st.sidebar.button("Result"):

            if prediction_option == "Player Prediction":
            
                predictions = make_predictions_player(clf_player, df, test)
                st.success("Result:")
                result=predictions[['PLAYER','PTS_bin','predict']]
                st.data_editor(result, num_rows="dynamic")
                csv_data_res = result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Үр дүнг татах",
                    data=csv_data_res,
                    file_name="result.csv",
                    key="download_result",
                )

                accuracy = accuracy_score(predictions['PTS_bin'],predictions['predict'])
                st.text(f"Accuracy: {accuracy:.2f}")
            progress_bar.progress(100)
       
if __name__ == '__main__':
    main()
