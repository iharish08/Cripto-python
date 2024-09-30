import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Load the data
csv_file_name = 'cpt.csv'
data = pd.read_csv(csv_file_name, parse_dates=['Date'], index_col='Date')

# Step 2: Check if data is loaded correctly
if data.empty:
    print("Data is empty. Please check the CSV file.")
else:
    # Step 3: Plot the historical prices
    plt.figure(figsize=(10, 6))
    plt.plot(data['Price'], label='Historical Price')
    plt.title('Historical Cryptocurrency Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Step 4: Fit the ARIMA model
    model = ARIMA(data['Price'], order=(5, 1, 0))  # Example: ARIMA(5,1,0)
    model_fit = model.fit()

    # Step 5: Display model summary
    print("\nARIMA Model Summary:")
    print(model_fit.summary())

    # Step 6: Forecasting
    forecast_steps = 30  # Number of steps to forecast
    forecast, stderr, conf_int = model_fit.forecast(steps=forecast_steps)

    # Step 7: Plotting the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Price'], label='Historical Price')
    plt.plot(pd.date_range(start=data.index[-1], periods=forecast_steps, freq='D'), forecast, label='Forecasted Price', color='red')
    plt.fill_between(pd.date_range(start=data.index[-1], periods=forecast_steps, freq='D'), conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
    plt.title('Cryptocurrency Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
