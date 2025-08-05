#マッピング辞書
weather_map={"晴れ":0, "くもり":1, "雨":2}

#「天気」列を数字に変換
df["天気"] = df["天気"].map(weather_map)