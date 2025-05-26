import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Process the overall data
def preprocess_stock_data(filepath = "F:/Tesla_project/Data/Tasla_Stock_Updated_V2.csv"):
    

    df = pd.read_csv(filepath)


    df['Target'] = df['Close'].shift(-1)

    df.dropna(inplace=True)

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    return df

# split the dataset
def split_stock_data(df, train_ratio=0.8, tail_extend=500):

    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
 
    if tail_extend > 0:
        train_df = pd.concat([train_df, train_df.tail(tail_extend)], axis=0)

    train_df = train_df.sort_index()
    test_df = test_df.sort_index()

    return train_df, test_df

# Preprocess the data for ml model
def preprocess_data_for_ml_model(train_df, test_df, features, target):


    # Extract features and target from train and test set
    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]
    
    # Create separate scalers for features and target
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit and transform the training data
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))

    # Transform the testing data using the same scalers
    X_test_scaled = feature_scaler.transform(X_test)
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler

# Preprocess the data for lstm model
def preprocess_for_lstm(train_df, test_df, features, target, time_steps=5):
    # Scale features and target for training
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features_train = feature_scaler.fit_transform(train_df[features])
    scaled_target_train = target_scaler.fit_transform(train_df[[target]])

    # Scale test features and target
    scaled_features_test = feature_scaler.transform(test_df[features])
    scaled_target_test = target_scaler.transform(test_df[[target]])

    # Create sequences
    def create_sequences(features, target, time_steps):
        X, y = [], []
        for i in range(time_steps, len(features)):
            X.append(features[i - time_steps:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    # Create sequences for training and test
    X_train_full, y_train_full = create_sequences(scaled_features_train, scaled_target_train, time_steps)
    X_test, y_test = create_sequences(scaled_features_test, scaled_target_test, time_steps)

    # Time-based validation split (10%)
    val_size = int(0.1 * len(X_train_full))
    X_val = X_train_full[-val_size:]
    y_val = y_train_full[-val_size:]

    X_train = X_train_full[:-val_size]
    y_train = y_train_full[:-val_size]

    # Return everything
    scalers = {'feature_scaler': feature_scaler, 'target_scaler': target_scaler}
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler,target_scaler










