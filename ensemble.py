# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:31:40 2020

@author: ER90614
"""
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pmdarima as pm
import LazyProphet as lp
import multiprocessing as mp
import warnings
warnings.simplefilter("ignore")

class ensemble:

    def __init__(self, series, forecast_horizon = 24, periods = 12):
        self.periods = periods
        self.forecast_horizon = forecast_horizon
        self.series = series
        
    def fit_ets(self, series):
        model = sm.tsa.ExponentialSmoothing(series,
                                            trend = 'add',
                                            damped = True,
                                            seasonal='mul',
                                            seasonal_periods=self.periods).fit()
        output = model.predict(start = 1, end = len(series) + self.forecast_horizon)
        fitted = output[:len(series)]
        predictions = output[-self.forecast_horizon:]
        
        return fitted, predictions
    
    def fit_naive(self, series):
        fitted = series
        predictions = np.resize(series[-1], self.forecast_horizon)
        
        return fitted, predictions

    def fit_mean(self, series):
        fitted = np.resize(np.mean(series), len(series))
        predictions = np.resize(np.mean(series), self.forecast_horizon)
        
        return fitted, predictions
    
    def fit_seasonal_naive(self, series):
        fitted = series
        predictions = np.resize(series[-self.periods:], self.forecast_horizon)
        
        return fitted, predictions
    
    
    def fit_arima(self, series):
        model = pm.auto_arima(series, start_p=1, start_q=1,
                               max_p=2, max_q=2, m=12,
                               start_P=0, seasonal=False,
                               d=1, D=1, trace=False,
                               error_action='ignore',  
                               suppress_warnings=True, 
                               stepwise=True)
        fitted = model.predict_in_sample(start=1, end=len(series))
        predictions = model.predict(self.forecast_horizon)
        
        return fitted, predictions
    
    def fit_lp(self, series):
        boosted_model = lp.LazyProphet(freq = self.periods, 
                                    estimator = 'ridge',
                                    max_boosting_rounds = 50,
                                    approximate_splits = True,
                                    regularization = 1.2,
                                    global_cost = 'mbic',
                                    split_cost = 'mse',
                                    )
        output = boosted_model.fit(series)
        fitted = output['yhat']
        predictions = boosted_model.extrapolate(self.forecast_horizon)
        
        return fitted, predictions
    
    def fit_mean_lp(self, series):
        boosted_model = lp.LazyProphet(freq = 0, 
                                    estimator = 'mean',
                                    max_boosting_rounds = 50,
                                    approximate_splits = True,
                                    regularization = 1.2,
                                    global_cost = 'mbic',
                                    split_cost = 'mse',
                                    seasonal_smoothing = 15
                                    )
        output = boosted_model.fit(series)
        fitted = output['yhat']
        predictions = boosted_model.extrapolate(self.forecast_horizon)
        
        return fitted, predictions
    
    
    def fit(self):
        output = {}
        output['lp_results'] = self.fit_lp(self.series)
        output['mean_lp_results'] = self.fit_mean_lp(self.series)
        output['arima_results'] = self..fit_arima(self.series)
        output['naive_results'] = self.fit_naive(self.series)
        output['ets_results'] = self.fit_ets(self.series)
        output['mean_results'] = self.fit_mean(self.series)
        output['seasonal_naive_results'] = self.fit_seasonal_naive(self.series)
        
        
        return output
    
    def multiprocess_fit(self, series):
        output = {}
        output['lp_results'] = self.fit_lp(series)
        output['mean_lp_results'] = self.fit_mean_lp(series)
        output['arima_results'] = self.fit_arima(series)
        output['naive_results'] = self.fit_naive(series)
        output['ets_results'] = self.fit_ets(series)
        output['mean_results'] = self.fit_mean(series)
        output['seasonal_naive_results'] = self.fit_seasonal_naive(series)
        
        return output
        
    def multiprocess(self):
        with mp.Pool(mp.cpu_count()-2) as p:
            results = list(tqdm(p.imap(self.multiprocess_fit, self.series)))
        
if __name__ == '__main__':       
    import quandl
    data = quandl.get("BITSTAMP/USD")
    
    y = data['Low']
    y = y[-730:]
    ens = ensemble(y, periods = 365)
    results = ens.fit()
    
    

