CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
mainから作業ブランチへのマージ
git fetch origin
git merge origin/main

精度の評価
R²スコア（決定係数）1に近いほど良い。予測と実測の一致度
MSE（平均二乗誤差）	小さいほど良い。誤差を2乗して平均
MAE（平均絶対誤差）	小さいほど良い。誤差の絶対値の平均

候補１.売上=β_0+β_1⋅雨フラグ+ε


候補2.晴れ → 平均売上高め
くもり → 平均売上中くらい
雨 → 平均売上下め
でも標準偏差（ばらつき）を持たせて、順序が逆転する日もある



class readdata
file_read 
openread (input_file)　名前
split_data 
 空白区切りでCsvファイルを分割　日付のデータをハイフン区切りで分割
convert_weather
入力値を（晴れ、くもり、雨）として、それぞれを0,1,2に変換する
input_data(weather,date)
予測したい日の日付と天気を入力 日付を空白区切りで入力

class learn data
学習する
予想結果print
