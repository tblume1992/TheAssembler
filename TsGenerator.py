# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:21:06 2020

@author: ER90614
"""
import matplotlib.pyplot as plt
import timesynth as ts
import random
import numpy as np
from tqdm import tqdm

class RandomTS:
    def __init__(self, min_samples, max_samples):
        self.num_points = random.randint(min_samples, max_samples)
        
        return
        
    def generate_sinusoidal(self):
        time_sampler = ts.TimeSampler()
        irregular_time_samples = time_sampler.sample_irregular_time(num_points=self.num_points)
        frequency = random.random()
        std = random.random()
        sinusoid = ts.signals.Sinusoidal(frequency=frequency)
        white_noise = ts.noise.GaussianNoise(std=std)
        timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
        samples, signals, errors = timeseries.sample(irregular_time_samples)
        if random.random() > .5:
            trend = random.uniform(-.01, .01)
            samples = samples + trend*np.array(range(self.num_points))
        
        return samples + 10

    def generate_psuedoperiodic(self):
        time_sampler = ts.TimeSampler()
        irregular_time_samples = time_sampler.sample_irregular_time(num_points=self.num_points)
        frequency = random.random()*2
        freqSD = random.uniform(.005, .1)
        pseudo_periodic = ts.signals.PseudoPeriodic(frequency = frequency, freqSD=freqSD, ampSD=0.5)
        timeseries_pp = ts.TimeSeries(pseudo_periodic)
        samples, signals, errors = timeseries_pp.sample(irregular_time_samples)
        if random.random() > .5:
            trend = random.uniform(-.01, .01)
            samples = samples + trend*np.array(range(self.num_points))
            
        return samples + 10
    
    def generate_car(self):
        time_sampler = ts.TimeSampler()
        irregular_time_samples = time_sampler.sample_irregular_time(num_points=self.num_points)
        ar_param = random.uniform(0, 1)
        sigma = random.uniform(0.01, .11)
        car = ts.signals.CAR(ar_param=ar_param, sigma=sigma)
        car_series = ts.TimeSeries(signal_generator=car)
        samples = car_series.sample(irregular_time_samples)[0]
            
        return samples + 10

    def generate(self):
        model = random.choice(['sinusoidal', 'psuedoperiodic', 'car', 'car', 'car'])
        if model == 'sinusoidal':
            samples = self.generate_sinusoidal()
        elif model == 'psuedoperiodic':
            samples = self.generate_psuedoperiodic()
        elif model == 'car':
            samples = self.generate_car()
            
        return samples
    
if __name__ == '__main__':
    for i in tqdm(range(1000)):
        random_series = RandomTS(760, 900).generate()
    plt.plot(random_series)
        

