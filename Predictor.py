from flask import Flask, request, jsonify, render_template
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from flask_cors import CORS
import datetime
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

model_rand = None
model_xgb = None
df = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload-csv", methods=["POST"])
def upload_csv():
    global df, model_rand, model_xgb

    file = request.files["file"]
    df = pd.read_csv(file)

    df["date"] = pd.to_datetime(df["日付"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].apply(lambda x: int(x in [0, 4, 5, 6]))

    weather_map = {"晴れ": 0, "くもり": 1, "雨": 2}
    df["weather"] = df["天気"].map(weather_map)
    df["sales"] = df["売り上げ"]

    X = df[["year", "month", "weather", "is_weekend"]]
    y = df["sales"]

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    def objective_rand(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return mean_squared_error(y_test, pred)

    def objective_xgb(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        }
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return mean_squared_error(y_test, pred)

    study_rand = optuna.create_study(direction="minimize")
    study_rand.optimize(objective_rand, n_trials=60)
    best_params_rand = study_rand.best_params
    best_params_rand["random_state"] = 42
    model_rand = RandomForestRegressor(**best_params_rand)
    model_rand.fit(X_train, y_train)

    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(objective_xgb, n_trials=30)
    best_params_xgb = study_xgb.best_params
    model_xgb = xgb.XGBRegressor(**best_params_xgb, random_state=42)
    model_xgb.fit(X_train, y_train)

    pred_rand = model_rand.predict(X_test)
    pred_xgb = model_xgb.predict(X_test)
    final_pred = 0.5 * pred_rand + 0.5 * pred_xgb

    r2 = r2_score(y_test, final_pred)
    mse = mean_squared_error(y_test, final_pred)
    mae = mean_absolute_error(y_test, final_pred)

    return jsonify({
        "message": "学習完了",
        "r2_score": round(r2, 4),
        "mse": round(mse, 2),
        "mae": round(mae, 2),
        "best_params_rand": best_params_rand,
        "best_params_xgb": best_params_xgb
    })

@app.route("/sendData", methods=["POST"])
def predict():
    global model_rand, model_xgb
    if model_rand is None or model_xgb is None:
        return jsonify({"error": "モデルが学習されていません"}), 400

    data = request.get_json()
    year = data["year"]
    month = data["month"]
    day = data["day"]
    weather = data["weather"]

    date = datetime.date(year, month, day)
    dayofweek = date.weekday()
    is_weekend = int(dayofweek in [0, 4, 5, 6])

    X_input = pd.DataFrame([[year, month, weather, is_weekend]],
                           columns=["year", "month", "weather", "is_weekend"])

    pred_rand = model_rand.predict(X_input)[0]
    pred_xgb = model_xgb.predict(X_input)[0]
    final_pred = 0.2 * pred_rand + 0.8 * pred_xgb

    return jsonify({"predicted_sales": int(final_pred)})

if __name__ == "__main__":
    app.run(debug=True)
