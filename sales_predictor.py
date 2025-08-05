import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

class read_data:
    def __init__(self, csv_path):
        """
        初期化：CSVファイルからデータを読み込み、モデルも準備
        """
        self.df = pd.read_csv(csv_path)
        self.model = RandomForestRegressor(random_state=42)

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
    start = time.time()
    loader = read_data("test.csv")  
    predictor = Learn_data(loader.df, loader.model)
    predictor.run()
    predictor.predict_single(2024, 8, 5, 0)
    end = time.time()
    run_time = end - start
    print(run_time)
