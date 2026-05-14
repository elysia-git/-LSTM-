from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/yichang_air.csv")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "yichang_monthly.csv"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    # 只保留需要的列
    df = df[["sj", "PM25", "PM10", "O3", "NO2", "SO2", "CO"]].copy()

    # 重命名成统一字段
    df.rename(columns={
        "sj": "time",
        "PM25": "pm25",
        "PM10": "pm10",
        "O3": "o3",
        "NO2": "no2",
        "SO2": "so2",
        "CO": "co"
    }, inplace=True)

    # 时间格式转换
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m", errors="coerce")

    # 转数字
    for col in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除空值
    df = df.dropna()

    # 按时间排序
    df = df.sort_values("time").reset_index(drop=True)

    # 保存处理后的训练数据
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print("✅ 数据处理完成")
    print(f"输出文件: {OUT_PATH}")
    print(df.head())
    print(f"总行数: {len(df)}")

if __name__ == "__main__":
    main()