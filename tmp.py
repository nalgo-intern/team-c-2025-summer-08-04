import pandas as pd
import numpy as np
import datetime

# Pandasの表示設定を変更して、全ての行を表示させる
pd.set_option('display.max_rows', None)

# 乱数シードを固定して、毎回同じ結果になるようにする
np.random.seed(123)

# 1. パラメータ設定
# ==========================================================================
# --- 基本設定 ---
YEARS = 5
start_date = datetime.date(2020, 1, 1)

# --- 天気関連 ---
transition_matrices = {
    'normal': pd.DataFrame({'晴れ': {'晴れ': 0.70, 'くもり': 0.25, '雨': 0.05}, 'くもり': {'晴れ': 0.35, 'くもり': 0.45, '雨': 0.20}, '雨':   {'晴れ': 0.30, 'くもり': 0.50, '雨': 0.20}}).T,
    'rainy_season': pd.DataFrame({'晴れ': {'晴れ': 0.50, 'くもり': 0.40, '雨': 0.10}, 'くもり': {'晴れ': 0.20, 'くもり': 0.50, '雨': 0.30}, '雨':   {'晴れ': 0.15, 'くもり': 0.50, '雨': 0.35}}).T,
    'summer': pd.DataFrame({'晴れ': {'晴れ': 0.80, 'くもり': 0.15, '雨': 0.05}, 'くもり': {'晴れ': 0.50, 'くもり': 0.35, '雨': 0.15}, '雨':   {'晴れ': 0.40, 'くもり': 0.40, '雨': 0.20}}).T,
    'winter': pd.DataFrame({'晴れ': {'晴れ': 0.65, 'くもり': 0.30, '雨': 0.05}, 'くもり': {'晴れ': 0.30, 'くもり': 0.50, '雨': 0.20}, '雨':   {'晴れ': 0.25, 'くもり': 0.50, '雨': 0.25}}).T
}
month_to_season_map = {1: 'winter', 2: 'winter', 3: 'normal', 4: 'normal', 5: 'normal', 6: 'rainy_season', 7: 'summer', 8: 'summer', 9: 'rainy_season', 10: 'normal', 11: 'normal', 12: 'winter'}

# --- 売上関連 ---
sales_stats = {
    '晴れ': {'mean': 160000, 'std': 8000},
    'くもり': {'mean': 140000, 'std': 7000},
    '雨':   {'mean': 120000, 'std': 6000}
}
seasonal_factors = {
    1: 0.95, 2: 0.95, 3: 1.0, 4: 1.0, 5: 1.05, 6: 1.05,
    7: 1.1, 8: 1.1, 9: 1.0, 10: 1.0, 11: 0.95, 12: 0.95
}
weekday_factors = {
    0: 1.0, 1: 0.95, 2: 0.95, 3: 1.0, 4: 1.1,
    5: 1.15, 6: 1.1
}

# ==========================================================================


# 2. データフレームの準備と天気データの生成 (変更なし)
# ==========================================================================
days = 365 * YEARS
dates = pd.date_range(start_date, periods=days, freq='D')
df = pd.DataFrame({'日付': dates})
months = df['日付'].dt.month
weathers = []
weather_states = list(transition_matrices['normal'].index)
current_weather = np.random.choice(weather_states)
weathers.append(current_weather)

for i in range(1, len(df)):
    current_month = months[i-1]
    season = month_to_season_map[current_month]
    current_transition_matrix = transition_matrices[season]
    transition_probabilities = current_transition_matrix.loc[current_weather]
    next_weather = np.random.choice(transition_probabilities.index, p=transition_probabilities.values)
    weathers.append(next_weather)
    current_weather = next_weather
df['天気'] = weathers
# ==========================================================================


# 3. 売上データの生成ループ (変更なし)
# ==========================================================================
sales = []
weekdays = df['日付'].dt.weekday
dates_for_event_check = df['日付'].dt.date

for i in range(len(df)):
    weather = weathers[i]
    month = months[i]
    weekday = weekdays[i]
    date = dates_for_event_check[i]

    base_mean = sales_stats[weather]['mean']
    expected_sale = base_mean * seasonal_factors[month]
    expected_sale *= weekday_factors[weekday]

    sale_target = expected_sale
    std_dev = sales_stats[weather]['std']
    final_sale = max(0, int(np.random.normal(loc=sale_target, scale=std_dev)))
    sales.append(final_sale)

df['売り上げ'] = sales
# ==========================================================================


# 4. 生成されたデータの確認とCSV保存
# ==========================================================================
print("---最終データセットの上から５行---")
print(df.head())
print("\n--- 売上統計情報 ---")
print(df['売り上げ'].describe())

print("\n--- 月ごとの平均天気日数（5年平均） ---")
monthly_weather_counts = df.groupby(df['日付'].dt.month)['天気'].value_counts().unstack(fill_value=0)
all_months_index = pd.Index(range(1, 13), name='月')
monthly_weather_counts_all = monthly_weather_counts.reindex(all_months_index, fill_value=0)
print(round(monthly_weather_counts_all / YEARS).astype(int))

df_for_csv = df.copy()
df_for_csv['日付'] = df_for_csv['日付'].dt.strftime('%Y-%m-%d')

# CSVファイルとして保存
csv_filename = 'sales_timeseries_data.csv'
df_for_csv.to_csv(csv_filename, encoding='utf-8-sig', index=False)
print(f"\nデータセットを '{csv_filename}' として保存しました。")