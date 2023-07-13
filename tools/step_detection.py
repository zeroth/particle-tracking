import os
import sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import curve_fit
# from mplcursors import cursor
from dataclasses import dataclass
import copy

# Auto step finder
import particle_tracking.stepfindCore as core
import particle_tracking.stepfindInOut as sio
import particle_tracking.stepfindTools as st

# reimplimenting AutoStepfinder
# https://www.cell.com/patterns/fulltext/S2666-3899(21)00082-9?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2666389921000829%3Fshowall%3Dtrue
def step_sim(step_size = 8.0, SNR = 0.5, total_time_point=400):
    np.random.seed(20181118)
    
    steps = np.piecewise(np.arange(0, total_time_point), 
                            [
                                np.arange(0, total_time_point) < 99, 

                                np.logical_and(np.arange(0, total_time_point) >= 99, np.arange(0, total_time_point) < 199), 

                                np.logical_and(np.arange(0, total_time_point) >= 199, np.arange(0, total_time_point) < 249), 
                                
                                np.logical_and(np.arange(0, total_time_point) >= 249, np.arange(0, total_time_point) < 349), 
                                
                                np.logical_and(np.arange(0, total_time_point) >= 349, np.arange(0, total_time_point) < 499), 

                                np.arange(0, total_time_point) >= 499
                            ], 
                            [
                                lambda x: 600, 
                                lambda x: 500, 
                                lambda x: 400 , 
                                lambda x: 300 ,
                                lambda x: 200 ,
                                lambda x: 100 ,
                            ])
    x = np.arange(0, total_time_point)
    noise_STD = step_size / SNR
    noise = np.random.normal(scale=noise_STD, size=len(x))
    # y = true_fct + np.random.standard_t(1, size=len(x))
    y = steps + noise
    return x, y, steps

def step_sim_v1(step_size = 8.0, SNR = 0.5, total_time_point=400):
    np.random.seed(20181118)
    
    steps = np.piecewise(np.arange(0, total_time_point), 
                            [
                                np.arange(0, total_time_point) < 99, 

                                np.logical_and(np.arange(0, total_time_point) >= 99, np.arange(0, total_time_point) < 199), 

                                np.logical_and(np.arange(0, total_time_point) >= 199, np.arange(0, total_time_point) < 249), 
                                
                                np.logical_and(np.arange(0, total_time_point) >= 249, np.arange(0, total_time_point) < 349), 
                                
                                np.logical_and(np.arange(0, total_time_point) >= 349, np.arange(0, total_time_point) < 499), 

                                np.arange(0, total_time_point) >= 499
                            ], 
                            [
                                lambda x: 300, 
                                lambda x: 100, 
                                lambda x: 50, 
                                lambda x: 350,
                                lambda x: 20,
                                lambda x: 350,
                            ])
    x = np.arange(0, total_time_point)
    noise_STD = step_size / SNR
    noise = np.random.normal(scale=noise_STD, size=len(x))
    # y = true_fct + np.random.standard_t(1, size=len(x))
    y = steps + noise
    return x, y, steps

def main():
    _, dataX, _ = step_sim(total_time_point=600)
    FitX = 0 * dataX

    # multipass:
    for ii in range(0, 3, 1):
        # work remaining part of data:
        residuX = dataX - FitX
        newFitX, _, _, S_curve, best_shot = core.stepfindcore(
            residuX, 0.1
        )
        FitX = st.AppendFitX(newFitX, FitX, dataX)
        # storage for plotting:
        if ii == 0:
            Fits = np.copy(FitX)
            S_curves = np.copy(S_curve)
            best_shots = [best_shot]
        elif best_shot > 0:
            Fits = np.vstack([Fits, FitX])
            S_curves = np.vstack([S_curves, S_curve])
            best_shots = np.hstack([best_shots, best_shot])

    # steps from final fit:
    steptable = st.Fit2Steps(dataX, FitX)
    print(steptable)
    # print(FitX)
    # print(steptable.shape)
    sio.SavePlot(
            ".", "test_2", dataX, Fits, S_curves, best_shots, steptable, 0.1
        )

if __name__ == "__main__":
    main()