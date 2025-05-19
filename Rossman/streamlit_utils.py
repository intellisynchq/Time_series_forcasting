import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


def plot_store_time_series(store_id, train_df):
    store_data = train_df[train_df['Store'] == store_id]
    
    # Group by date and sum sales
    sales_by_date = store_data.groupby('Date')['Sales'].sum().sort_index()
    x = np.arange(len(sales_by_date))
    y = sales_by_date.values

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(x, y)
    ax.set_title(f'Sales by Date for Store {store_id}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.grid(True)

    return fig


def identify_outliers(train_df):
    df = train_df.copy()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Date'])
    window = 7

    df['rolling_mean'] = df.groupby('Store')['Sales'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    df['rolling_std'] = df.groupby('Store')['Sales'].transform(lambda x: x.rolling(window=window, min_periods=1).std())

    # Avoid division by zero (std can be 0 in early days)
    df['rolling_std'] = df['rolling_std'].replace(0, np.nan)

    df['z_score'] = (df['Sales'] - df['rolling_mean']) / df['rolling_std']
    df['is_outlier'] = df['z_score'].abs() > 2
    df.reset_index(drop=True, inplace=True)

    outliers = df[df['is_outlier'] == True]
    print(outliers[['Date', 'Store', 'Sales', 'z_score']])

    return df


def plot_outliers(store_id, df):
    store_data = df[df['Store'] == store_id].copy()
    fig, ax = plt.subplots(figsize=(15, 5))
    
    ax.plot(store_data['Date'], store_data['Sales'], label='Sales')
    
    # Plot outliers (z_score > 2 or < -2)
    outlier_mask = store_data['is_outlier']
    ax.scatter(store_data.loc[outlier_mask, 'Date'], 
               store_data.loc[outlier_mask, 'Sales'], 
               color='red', label='Outlier (|z-score| > 2)')
    
    ax.legend()
    ax.set_title(f"Store {store_id} - Sales with Outliers Highlighted")
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    
    return fig



def prediction(final_df):
    feature_cols = [
        'Promo', 'Promo2', 'SchoolHoliday', 'DayOfWeek',
        'CompetitionDistance',
        'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14',
        'Sales_ma_7', 'Sales_ma_14'
    ]

    cols_to_drop = ['Assortment', 'StoreType', 'StateHoliday']

    df_model = final_df.dropna(subset=feature_cols +  cols_to_drop + ['Sales'])
    df_model['Sales_target'] = df_model.groupby('Store')['Sales'].shift(-1)

    # Drop last day of each store (no target for next day)
    df_model = df_model.dropna(subset=['Sales_target'])

    # Optional: encode categorical features
    df_model = pd.get_dummies(df_model, columns=['StoreType', 'Assortment', 'StateHoliday'])

    print(final_df['Date'])
    train = df_model[df_model['Date'] < '2015-03-01']
    test = df_model[df_model['Date'] >= '2015-03-01']

    X_train, y_train = train[feature_cols], train['Sales_target']
    X_test, y_test = test[feature_cols], test['Sales_target']

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )

    model.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False)
    
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # Ensure predictions are non-negative
    return y_test, y_pred


def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_test.values, label='Actual Sales')
    ax.plot(y_pred, label='Predicted Sales')
    ax.set_title('Actual vs Predicted Sales')
    ax.set_xlabel('Sample Index (time ordered)')
    ax.set_ylabel('Sales')
    ax.legend()
    return fig
