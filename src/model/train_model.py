import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam

def train_models_only(X_train, y_train, save_path):

    # Initialize models
    model_lr = LinearRegression()
    model_dt = DecisionTreeRegressor(random_state=42)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train models
    model_lr.fit(X_train, y_train)
    model_dt.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)

    # Save models
    joblib.dump(model_lr, f"{save_path}/model_lr.pkl")
    joblib.dump(model_dt, f"{save_path}/model_dt.pkl")
    joblib.dump(model_rf, f"{save_path}/model_rf.pkl")

    return model_lr, model_dt, model_rf

def build_sequence_model(model_type, input_shape):
 
    if model_type.lower() == 'lstm':
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
    elif model_type.lower() == 'tcn':
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
            MaxPooling1D(pool_size=3),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
    else:
        raise ValueError("Invalid model_type. Choose 'lstm' or 'tcn'.")

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Training LSTM-TCN model

def train_sequence_model(model, model_name, X_train, y_train, X_val, y_val, save_path, epochs, batch_size):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0
    )

    # Save in modern Keras format
    model.save(f"{save_path}/model_{model_name}.keras")

    print(f"Training complete for {model_name} model")
    return history





