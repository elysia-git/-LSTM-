from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

MODEL_PATH = "lstm_model.keras"
SCALER_X_PATH = "scaler_x.pkl"
SCALER_Y_PATH = "scaler_y.pkl"

LOOKBACK = 24
FEATURE_COUNT = 11

# 加载模型和归一化器
model = load_model(MODEL_PATH)
scaler_x = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

class InputData(BaseModel):
    values: list[list[float]]

@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.values, dtype=np.float32)

    # 校验输入形状
    if arr.shape != (LOOKBACK, FEATURE_COUNT):
        raise HTTPException(
            status_code=400,
            detail=f"输入形状必须是 ({LOOKBACK}, {FEATURE_COUNT})，当前是 {arr.shape}"
        )

    # 归一化
    arr_scaled = scaler_x.transform(arr)

    # 变成 LSTM 输入格式: (1, 24, 11)
    X = arr_scaled.reshape(1, LOOKBACK, FEATURE_COUNT)

    # 预测
    pred_scaled = model.predict(X, verbose=0)

    # 反归一化
    pred = scaler_y.inverse_transform(pred_scaled)[0, 0]

    return {
        "prediction": float(pred)
    }