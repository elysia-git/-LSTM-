import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = Path("data/processed/beijing_aq.csv")
MODEL_PATH = Path("lstm_model.keras")
SCALER_X_PATH = Path("scaler_x.pkl")
SCALER_Y_PATH = Path("scaler_y.pkl")

# 先选一个站点训练，最稳
STATION_NAME = "Aotizhongxin"

# 用过去24小时预测下一小时
LOOKBACK = 24

FEATURES = ["pm25", "pm10", "so2", "no2", "co", "o3", "temp", "pres", "dewp", "rain", "wspm"]
TARGET = "pm25"

def make_windows(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback])
    return np.array(Xs), np.array(ys)

def main():
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # 只选一个站点
    df = df[df["station"] == STATION_NAME].copy()
    df = df.dropna().sort_values("time").reset_index(drop=True)

    print(f"站点: {STATION_NAME}")
    print(f"样本数: {len(df)}")

    if len(df) < LOOKBACK + 1:
        raise ValueError(f"数据太少，至少需要 {LOOKBACK+1} 条，当前只有 {len(df)} 条")

    X_raw = df[FEATURES].astype(np.float32).values
    y_raw = df[[TARGET]].astype(np.float32).values

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    X, y = make_windows(X_scaled, y_scaled, LOOKBACK)

    print(f"X.shape = {X.shape}, y.shape = {y.shape}")

    # 训练集 / 验证集
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")

    model = Sequential([
        LSTM(64, input_shape=(LOOKBACK, len(FEATURES))),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(MODEL_PATH)
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    print("✅ 模型训练完成")
    print(f"模型文件: {MODEL_PATH}")
    print(f"输入归一化器: {SCALER_X_PATH}")
    print(f"输出归一化器: {SCALER_Y_PATH}")

if __name__ == "__main__":
    main()