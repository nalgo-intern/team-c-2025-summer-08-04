from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from flask_cors import CORS
import datetime
app = Flask(__name__)
CORS(app)

model = None
df = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload-csv", methods=["POST"])
def upload_csv():
    global df, model
    file = request.files["file"]
    df = pd.read_csv(file)

    df["date"] = pd.to_datetime(df["日付"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek  
    df["is_weekend"] = df["dayofweek"].apply(lambda x: int(x in [0, 4, 5, 6]))

    weather_map = {"晴れ": 0, "くもり": 1, "雨": 2}
    df["weather"] = df["天気"].map(weather_map)
    df["sales"] = df["売り上げ"]

    X = df[["year", "month", "day", "weather","is_weekend"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(max_depth=9,random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return jsonify({
        "message": "CSVを受け取り、学習と評価が完了しました",
        "r2_score": round(r2, 4),
        "mse": round(mse, 2),
        "mae": round(mae, 2)
    })

@app.route("/sendData", methods=["POST"])
def predict():
    global model
    if model is None:
        return jsonify({"error": "モデルが学習されていません"}), 400

    data = request.get_json()
    year = data["year"]
    month = data["month"]
    day = data["day"]

    date = datetime.date(year, month, day)
    weather = data["weather"]
    dayofweek = date.weekday()  
    is_weekend = int(dayofweek in [0, 4, 5, 6])

    X_input = pd.DataFrame([[year, month, day, weather, is_weekend]],
                           columns=["year", "month", "day", "weather", "is_weekend"])

    pred = model.predict(X_input)[0]
    return jsonify({"predicted_sales": int(pred)})

if __name__ == "__main__":
    app.run(debug=True)
