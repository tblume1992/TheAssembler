# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:36:29 2020

@author: ER90614
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy, variation
import statsmodels.api as sm
import LazyProphet as lp
import inspect
import itertools

class FeatureExtraction:
    def __init__(self, series, period):
        self.series = pd.Series(np.reshape(series, (-1, ))).reset_index(drop = True)
        self.period = period
        self.extract()
        
        return
        
    def get_differenced_series(self):
        self.diff_series = self.series.diff().dropna()
        self.diff_2_series = self.diff_series.diff().dropna()
        self.seasonal_diff_series = self.diff_series.diff(self.period).dropna()
        
        return 
    
    def get_acf(self):
        self.acf = sm.tsa.stattools.acf(self.series, nlags = 10)
        self.diff_acf = sm.tsa.stattools.acf(self.diff_series, nlags = 10)
        self.diff_2_acf = sm.tsa.stattools.acf(self.diff_2_series, nlags = 10)
        self.seasonal_acf = sm.tsa.stattools.acf(self.seasonal_diff_series)
        
        return
    
    def get_pacf(self):
        self.pacf = sm.tsa.stattools.pacf(self.series, nlags = 10)
        self.diff_pacf = sm.tsa.stattools.pacf(self.diff_series, nlags = 10)
        self.diff_2_pacf = sm.tsa.stattools.pacf(self.diff_2_series, nlags = 10)
        self.seasonal_pacf = sm.tsa.stattools.pacf(self.seasonal_diff_series)
        
        return
    
    def get_lazy_prophet_output(self):
        self.lp_model = lp.LazyProphet(freq = self.period, 
                                       estimator = 'ridge', 
                                       poly = 2,
                                       global_cost = 'maicc',
                                       split_cost = 'mse')
        self.lp_output = self.lp_model.fit(self.series)
        self.lp_coefs = self.lp_model.coefs[0]
        self.lp_deseasonalized = self.lp_output['y'] - self.lp_output['seasonality']
        self.lp_remainder = self.lp_output['y'] - self.lp_output['yhat']
        
        
        return
        
    
    def calc_hurst(self):
        series = np.array(self.series)
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        return poly[0]*2.0
       
    def calc_entropy(self):
        return entropy(self.series)
    
    def calc_acf_10(self):
        return np.sum(self.acf[1:11]**2)
    
    def calc_acf_diff_10(self):
        return np.sum(self.diff_acf[1:11]**2)
    
    def calc_acf(self):
        return self.acf[1]
    
    def calc_acf_diff(self):
        return self.diff_acf[1]
    
    def calc_acf_2_diff_10(self):
        return np.sum(self.diff_2_acf[1:11]**2)
    
    def calc_acf_2_diff(self):
        return self.diff_2_acf[1]
    
    def calc_seasonal_acf(self):
        return self.seasonal_acf[1]
    
    def calc_pacf_10(self):
        return np.sum(self.pacf[1:11]**2)
    
    def calc_pacf_diff_10(self):
        return np.sum(self.diff_pacf[1:11]**2)
    
    def calc_pacf(self):
        return self.pacf[1]
    
    def calc_pacf_diff(self):
        return self.diff_pacf[1]
    
    def calc_pacf_2_diff_10(self):
        return np.sum(self.diff_2_pacf[1:11]**2)
    
    def calc_pacf_2_diff(self):
        return self.diff_2_pacf[1]
    
    def calc_seasonal_pacf(self):
        return self.seasonal_pacf[1]
    
    def calc_crossing_points(self):
        crossing_points = len(list(itertools.groupby(self.series-np.median(self.series),
                                                     lambda Input: Input > 0))) 
        return crossing_points - 1
    
    def calc_seasonal_period(self):
        return self.period
    
    def calc_nperiods(self):
        return int(len(self.series)/self.period)
    
    def calc_lp_curvature(self):
        return self.lp_coefs[1]
    
    def calc_lp_linearity(self):
        return self.lp_coefs[0]
    
    def calc_cov(self):
        return variation(self.series)
    
    def calc_lp_trend(self):
        lp_trend = np.std(self.lp_remainder)**2 / np.std(self.lp_deseasonalized)**2
        
        return max((0, 1 - lp_trend))
    
# =============================================================================
#     def calc_lp_spike(self):
#         leave_one_out = [self.lp_remainder.drop(i).var() for i in range(len(self.lp_remainder))]
#         
#         return np.std(leave_one_out)
# =============================================================================
    
    def extract(self):
        self.features = {}
        self.get_differenced_series()
        self.get_acf()
        self.get_pacf()
        self.get_lazy_prophet_output()
        for method in inspect.getmembers(FeatureExtraction, predicate=inspect.isfunction):
            if 'calc' in method[0]:
                self.features[method[0]] = method[1](self)
                
        return 
        
 # %%   
if __name__ == '__main__':
    import quandl
    data = quandl.get("BITSTAMP/USD")
    y = data['Low']
    features = FeatureExtraction(y, period = 365).features


