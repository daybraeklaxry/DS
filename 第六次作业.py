import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_and_preprocess_data(file_path):
    dataframe = pd.read_csv(file_path)
    dataframe = dataframe.drop(columns=['id'])

    dataframe = dataframe[dataframe['city'] == 1]
    dataframe = dataframe.drop(columns=['city'])

    dataframe.loc[dataframe['hour'] >= 19, 'hour'] = 0
    dataframe.loc[dataframe['hour'] <= 5, 'hour'] = 0
    dataframe.loc[dataframe['hour'] != 0, 'hour'] = 1
    
    return dataframe

def prepare_features_and_target(dataframe):
    target = dataframe['y'].values.reshape(-1, 1)
    features = dataframe.drop(columns=['y']).values
    return features, target

def scale_data(features, target):
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    target_scaled = scaler.fit_transform(target)
    return features_scaled, target_scaled, scaler

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    predictions = regressor.predict(X_test)
    
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse

def main():
    file_path = 'bike.csv'
    
    dataframe = load_and_preprocess_data(file_path)
    
    features, target = prepare_features_and_target(dataframe)
    
    X_scaled, y_scaled, scaler = scale_data(features, target)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    rmse = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

if __name__ == "__main__":
    main()
