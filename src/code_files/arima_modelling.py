def get_arima_forecasts(data,column_name, p,d,q, train_fraction):
    '''
    Predicts the forecast using the ARIMA model
    :param data:
    :param column_name:
    :param p: parameter for lags
    :param d: parameter for difference
    :param q: parameter for moving average
    :return: forecasted values
    '''
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error

    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(p, int) and isinstance(d, int) and isinstance(q, int)
    assert isinstance(train_fraction, float) and train_fraction > 0 and train_fraction <1

    # split the data in train-test
    train_size = int(len(data) * train_fraction)
    train_data, test_data = data[:train_size], data[train_size:]

    # Fit the model to training data. Replace p, d, q with our ARIMA parameters
    model = ARIMA(train_data[column_name], order=(p, d, q))

    # Forecast
    model_fit = model.fit()
    forecast_test_data = model_fit.forecast(steps=len(test_data))

    rmse = mean_squared_error(test_data[column_name], forecast_test_data, squared=False)
    print(f"RMSE: {rmse}")

    predictions = [np.nan] * train_size + forecast_test_data.values.tolist()
    data['predictions'] = predictions
    return data