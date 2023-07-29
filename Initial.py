## dataset mango-no-lag
df = pd.read_csv('/home/hossein//mango_grouped.csv', sep=";")
data3=df
data3.loc[83,"KilosEntrados"]=0
data3.loc[107,"KilosEntrados"]=0
data3.loc[134,"KilosEntrados"]=0

## dataset mango-l24
# df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l4.csv', sep=";")

df = pd.read_csv('/home/hossein//differenced_mango_l24.csv', sep=";")
# df = df.drop(df.index[0])
data3=df
data3.index = pd.to_datetime(data3['FechaRecepcion'])
data3.drop(columns='FechaRecepcion',inplace=True)






# df = pd.read_csv('/home/hossein//mango_l24.csv', sep=";")
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l4.csv', sep=";")

df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
# selected_features = ['KilosEntrados', 'FechaRecepcion']+ E
selected_features = ['KilosEntrados', 'FechaRecepcion']+ E.tolist()
# selected_features = ['KilosEntrados', 'FechaRecepcion']+ E

data3=df[selected_features]
# data3 = df[['KilosEntrados','FechaRecepcion',  'tmedia']]
# data3.drop(data3.tail(24).index,inplace=True) # drop last n rows

# data3=df


data3.index = pd.to_datetime(data3['FechaRecepcion'])
data3.drop(columns='FechaRecepcion',inplace=True)




## dataset mango-l12
df = pd.read_csv('/home/hossein/Downloads/avocado_datasets/avocado_l12.csv', sep=";")

# df = pd.read_csv('/home/hossein/mango_l4.csv', sep=";")
data3 = df[['KilosEntrados','FechaRecepcion','dir (t-1)']]
data3['KilosEntrados'] = data3['KilosEntrados'].clip(lower=0)
# data3.drop(data3.tail(12).index,inplace=True) # drop last n rows
# data3=df
data3.index = pd.to_datetime(data3['FechaRecepcion'])
data3.drop(columns='FechaRecepcion',inplace=True)

#### avocado-l12

df = pd.read_csv('/home/hossein/no_lag_avocado.csv', sep=";")
df['KilosEntrados'] = df['KilosEntrados'].clip(lower=0)
data3 = df[['KilosEntrados','FechaRecepcion','Superficie Cultivo (t-8)', 'tmin (t-7)']]
# data3=df
data3.index = pd.to_datetime(data3['FechaRecepcion'])
data3.drop(columns='FechaRecepcion',inplace=True)

# data3.drop(columns='dir',inplace=True)
# data3.drop(columns='prec',inplace=True)
# data3.drop(columns='velmedia',inplace=True)
# data3.drop(columns='tmin',inplace=True)
# data3.drop(columns='hmedia',inplace=True)
# data3.drop(columns='NArboles',inplace=True)
# data3.drop(columns='tmedia',inplace=True)
# # data3.drop(columns='tmax',inplace=True)
# data3.drop(columns='Superficie Cultivo',inplace=True)

# data3['KilosEntrados'].plot()


df_train = data3[:-12]
df_test = data3[-12:]
df_train=df_train.reset_index()
df_test=df_test.reset_index()
df_train.drop(columns='FechaRecepcion',inplace=True)
df_test.drop(columns='FechaRecepcion',inplace=True)




def sarimax_evaluate_models(df1,df2, configs):
    """
    Evaluates all possible SARIMAX parameters.

    Returns the best parameter selection.
    """
    best_cfg = None
    best_score=-100000

    for config in configs:
        p, d, q, P_value, D_value, Q_value, seasonality = config
        order = (p, d, q)
        s_order = (P_value, D_value, Q_value, seasonality)
        try:
            r2 = evaluate_sarimax_model_r2(df1,df2, order, s_order)
            print("Configuration: ", config, " has r2 of: ", r2)
            if r2 > best_score:
                best_score, best_cfg = r2, [order, s_order]
        except Exception as err:
            print(f"SARIMAX config {config} has raised an error.")
            print(err)
            continue

    return best_cfg

def evaluate_sarimax_model_r2(df1, df2, order, s_order):
    """
    Evaluates the SARIMAX model given certain parameters using R2.

    Returns the R2 score as a float value
    """
    try:
        model = SARIMAX(
            endog=df1["KilosEntrados"],
            exog=df1.drop(["KilosEntrados"], axis=1),
            # endog=df1.iloc[:,df1.shape[1]-1],
            # exog=df1.iloc[:,0:df1.shape[1]-1],
            order=order,
            seasonal_order=s_order,
            enforce_invertibility=False,
            enforce_stationarity=False,

        )
        results = model.fit(disp=False)

        pred_uc = results.get_forecast(
            exog=df2.drop(["KilosEntrados"], axis=1), steps=12
        )
        # pred_uc = results.get_forecast(
        #     exog=df2.iloc[:,0:df2.shape[1]-1], steps=12
        # )
        
        return r2_score(df2["KilosEntrados"].values, pred_uc.predicted_mean)
        # return r2_score(df2.iloc[:,df2.shape[1]-1], pred_uc.predicted_mean)
    except Exception as err:
        print(err)
        return float("inf")
    
    
def sarimax_configs(seasonality: int = 12):
    """
    Method to get all posible parameter configurations for Grid Search.

    Returns a list of tuples with parameters
    """
    p_params = [0, 1, 2]
    d_params = [1]
    q_params = [0, 1, 2]
    P_params = [0, 1, 2]
    D_params = [1]
    Q_params = [0, 1, 2]

    configs = product(
        p_params, d_params, q_params, P_params, D_params, Q_params, [seasonality]
    )

    return configs

from itertools import product
import time
# search space for the grid search: all possible configurations
from statsmodels.tsa.statespace.sarimax import SARIMAX
configs=sarimax_configs()
# grid search
t0=time.time()
best_cfg = sarimax_evaluate_models(df_train,df_test, configs)
t1=time.time()
print(t1-t0)
def sarimax_fit(df, config):
    """
    Method to train SARIMA given data and its parameters.

    Returns the fitted model.
    """
    order, sorder = config

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = SARIMAX(
            endog=df["KilosEntrados"],
            # endog=df.iloc[:,df.shape[1]-1],
            exog=df.drop(["KilosEntrados"], axis=1),
            # exog=df.iloc[:,0:df.shape[1]-1]
            order=order,
            seasonal_order=sorder,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit_model = model.fit(disp=False)

    return fit_model

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


results = sarimax_fit(df_train, best_cfg)
forecast=results.get_forecast(exog=df_test.drop(["KilosEntrados"], axis=1),steps=12)
forecasted_values = forecast.predicted_mean

ax = df_test["KilosEntrados"].plot(label="Real values")
forecasted_values.plot(
    ax=ax, label="Predicted values", alpha=0.7, figsize=(14, 7)
)

ax.set_xlabel("Date")
ax.set_ylabel("Kilos")
plt.legend()

plt.show()


import math
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def theil_index(y, y_est):
    n = len(y)
    num = math.sqrt(np.sum(np.power(y - y_est, 2)) / n)
    den1 = math.sqrt(np.sum(np.power(y, 2)) / n)
    den2 = math.sqrt(np.sum(np.power(y_est, 2)) / n)
    return num / (den1 + den2)


mae_sarimax = round(mean_absolute_error(df_test["KilosEntrados"].values, forecasted_values))
rmse_sarimax = round(math.sqrt(mean_squared_error(df_test["KilosEntrados"].values, forecasted_values)))
r2_sarimax = round(r2_score(df_test["KilosEntrados"].values, forecasted_values),2)
theil_sarimax = (theil_index(df_test["KilosEntrados"].values, forecasted_values))
