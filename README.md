# Time series forcasting

VistaMart Collective is a regional chain of retail stores offering a wide range of consumer goods, from household essentials to seasonal items. To support inventory planning, budgeting, and strategic decision-making, this task involves developing a time series forecasting model that predicts future sales across the VistaMart network.

Using historical sales data collected from multiple store locations over time, the goal is to build a robust forecasting model capable of accurately estimating future sales trends. The model should account for patterns such as seasonality, promotional events, and store-specific variations.

Key objectives include:
- Cleaning and preprocessing time series data from various store locations.
- Feature engineering: lag features, moving averages, rolling statistics.
- Exploring trends, seasonal cycles, and anomalies in historical sales.
- Building and evaluating forecasting models (e.g., ARIMA, Prophet, XGBoost or machine learning-based approaches).

The final model will serve as a foundation for data-driven decision-making at VistaMart Collective, helping the business anticipate demand.

## Data overview

You are provided with data for 1115 stores. The task is to forecast the "Sales" column using previous data.

Data is split into three files:
- dataset.csv: cointains daily information about store sales and customers
- promotions.csv: contains information about ongoing promotions
- stores.csv: contains general information about each store

A brief explanation of columns:
- Sales: the income for any given day (this is what you are predicting)
- Customers: the number of customers on a given day
- Assortment: describes an assortment level (small, medium or large)
- CompetitionDistance - distance in meters to the nearest competitor store
- CompetitionOpenSince[Month/Year] - gives the year and month of the time the nearest competitor was opened
- Promo: indicates whether a store is running a promo
- PromoSince[Year/Week]: describes the year and week when the store started participating in Promo
- PromoInterval: describes the consecutive intervals Promo is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

## What the Reviewer Will Look For
- Logical, clean approach to problem-solving.
- Clear EDA and visual storytelling.
- Justified modeling choices with error metrics.
- Thoughtful feature engineering.
- Clear documentation and reproducibility (README, Jupyter notebooks or scripts)

## Deliverables
 A GitHub repo with:
- Code (Jupyter notebooks or .py files)
- Dataset or link to it
- README with project overview and findings
