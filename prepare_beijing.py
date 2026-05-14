from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "beijing_aq.csv"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_files = list(RAW_DIR.rglob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"在 {RAW_DIR} 下没有找到 csv 文件，请检查解压路径。")

    print("找到文件数：", len(all_files))

    dfs = []
    for file in all_files:
        df = pd.read_csv(file)

        df["time"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]],
            errors="coerce"
        )

        keep_cols = [
            "time", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
            "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "station"
        ]
        df = df[keep_cols].copy()

        df.rename(columns={
            "PM2.5": "pm25",
            "PM10": "pm10",
            "SO2": "so2",
            "NO2": "no2",
            "CO": "co",
            "O3": "o3",
            "TEMP": "temp",
            "PRES": "pres",
            "DEWP": "dewp",
            "RAIN": "rain",
            "WSPM": "wspm"
        }, inplace=True)

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    num_cols = ["pm25", "pm10", "so2", "no2", "co", "o3", "temp", "pres", "dewp", "rain", "wspm"]
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["time"])
    data = data.sort_values(["station", "time"]).reset_index(drop=True)

    data[num_cols] = (
        data.groupby("station")[num_cols]
        .transform(lambda x: x.interpolate(limit_direction="both").ffill().bfill())
    )

    data.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print("✅ 数据处理完成")
    print(f"输出文件：{OUT_FILE}")
    print(data.head())
    print(f"总行数：{len(data)}")
    print(f"站点数：{data['station'].nunique()}")

if __name__ == "__main__":
    main()