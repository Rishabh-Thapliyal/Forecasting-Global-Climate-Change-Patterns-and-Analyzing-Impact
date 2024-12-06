import pandas as pd


def get_lgbm_model(df):
    '''
    get predictions using lgbm model
    :param df: pandas dataframe
    :return: np.array
    '''
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import lightgbm as lgb
    
    assert isinstance(df, pd.DataFrame)

    train = df.loc[(df["year"] < 2015), :]

    val = df.loc[(df["year"] >= 2015), :]

    cols = [col for col in train.columns if col not in ["year", "Actual Temperature_Air_Global"]]

    Y_train = train['Actual Temperature_Air_Global']
    X_train = train[cols]
    Y_val = val['Actual Temperature_Air_Global']
    X_val = val[cols]

    params = {
        'n_estimators': 2000,
        'max_depth': 4,
        'num_leaves': 2 ** 4,
        'learning_rate': 0.1,
        'boosting_type': 'dart'
    }

    model = lgb.LGBMRegressor(first_metric_only=True, **params)

    pipe = make_pipeline(StandardScaler(), model)

    pipe.fit(X_train, Y_train)

    # predicting values
    y_pred = pipe.predict(X_val)

    return y_pred

def lag_features(dataframe, lags):
    '''
    create rolling mean features
    :param dataframe: pandas dataframe
    :param lags: list of ints
    :return: dataframe with lag features
    '''
    import pandas as pd
    assert isinstance(dataframe, pd.DataFrame) and isinstance(lags, list)

    for lag in lags:
        dataframe['temp_lag_' + str(lag)] = dataframe.groupby(["Latitude", "Longitude"])['Actual Temperature_Air_Global'].transform(
            lambda x: x.shift(lag))
    return dataframe

def roll_mean_features(dataframe, windows):
    '''
    create rolling mean features
    :param dataframe: pandas dataframe
    :param windows: list of ints
    :return: dataframe with rolling mean features
    '''
    assert isinstance(dataframe, pd.DataFrame) and isinstance(windows, list)

    for window in windows:
        dataframe['temp_roll_mean_' + str(window)] = dataframe.groupby(["Latitude", "Longitude"])['Actual Temperature_Air_Global'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=0,).mean())
    return dataframe


def ewm_features(dataframe, alphas, lags):
    '''
    create exponential weighted mean features
    :param dataframe: pandas dataframe
    :param alphas: float
    :param lags: list of ints
    :return: dataframe with exponential weighted mean features
    '''

    assert isinstance(dataframe, pd.DataFrame) and isinstance(lags, list) and isinstance(alphas, float)

    for alpha in alphas:
        for lag in lags:
            dataframe['temp_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["Latitude", "Longitude"])['Actual Temperature_Air_Global'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe