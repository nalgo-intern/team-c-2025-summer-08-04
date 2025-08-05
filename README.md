売上=β_0+β_1⋅雨フラグ+ε



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
