import pandas as pd

class read_data():
    def __init__(self, file_path):
        """
        ファイルの読み込み
        """
        self.df = pd.read_csv(file_path)

    def split_data(self):
        """
        年月日の分割
        """
        # 各列の変数は仮の値
        self.df["year"] = None
        self.df["month"] = None
        self.df["day"] = None
        # 日付のデータが0列目にある想定
        for i in range(len(self.df)):
            Y, M, D = self.df.iloc[i, 0].split("-")
            self.df.iloc[i, 3] = Y
            self.df.iloc[i, 4] = M
            self.df.iloc[i, 5] = D