import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

#マージテスト

"""
後で変更した内容を統合
"""
class read_data:
    def __init__(self, csv_path):
        """
        初期化：CSVファイルからデータを読み込み、モデルも準備
        """
        self.df = pd.read_csv(csv_path)

        # 列名変換
        self.df.rename(columns={
            "日付": "date",
            "売り上げ": "sales",
            "天気": "weather"
        }, inplace=True)

        # 日付を年月日に分解
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["day"] = self.df["date"].dt.day

        # 天気を数値に変換
        weather_map = {"晴れ": 0, "くもり": 1, "雨": 2}
        self.df["weather"] = self.df["weather"].map(weather_map)

        self.model = RandomForestRegressor(random_state=42)
    def split_data(self):
        for i in range(len(self.df)):
            dt = self.df.iloc[i, 0]  
            Y, M, D = dt.year, dt.month, dt.day
    def input_data(self):
        pre_year,pre_month,pre_day= input("空白区切りで日付を入力してください(year month day):").split()
        weather_str = input("天気を入力してください:(晴れ くもり 雨):")
        weather_map={"晴れ":0, "くもり":1, "雨":2}
        pre_weather = weather_map[weather_str]
        return int(pre_year),int(pre_month),int(pre_day),pre_weather

class Learn_data:
    def __init__(self, df, model):
        self.df = df
        self.model = model

    def prepare_data(self):
        """
        説明変数と目的変数に分けて、学習用・テスト用に分割
        """
        X = self.df[["year", "month", "day", "weather"]]
        y = self.df["sales"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train):
        """
        モデルの学習
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        テストデータによる予測と評価
        """
        y_pred = self.model.predict(X_test)
        print("R^2 score:", r2_score(y_test, y_pred))
        print("MSE:", mean_squared_error(y_test, y_pred))

    def predict_single(self, year, month, day, weather):
        """
        任意の入力（1件）の売上予測
        """
        input_df = pd.DataFrame([{
            "year": year,
            "month": month,
            "day": day,
            "weather": weather
        }])
        prediction = self.model.predict(input_df)
        print(f"予測売上（{year}/{month}/{day} 天気={weather}）: {prediction[0]:.0f}円")
        return prediction[0]

    def run(self):
        """
        学習から評価まで一括実行
        """
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.train(X_train, y_train)
        self.evaluate(X_test, y_test)

if __name__ == "__main__":
    loader = read_data("test.csv") 
    pre_year,pre_month,pre_day,pre_weather = loader.input_data() 
    loader.split_data()
    start = time.time()
    predictor = Learn_data(loader.df, loader.model)
    predictor.run()
    predictor.predict_single(pre_year,pre_month,pre_day,pre_weather)
    end = time.time()
    run_time = end - start
    print(run_time)
