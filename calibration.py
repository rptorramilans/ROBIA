"""
Created on Fri Oct 10 2025 ‏‎‏‎17:59:38
Code part of ROBIA (UPC-BarcelonaTech)
Last modified on Fri Nov 28 2025
@author: raquel peñas torramilans
@contact: raquel.penas@upc.edu
"""

import os
import pandas as pd
import numpy as np
import json
import rasterio
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import savemat
from pathlib import Path
import params as p

def calibrate():

    data = pd.read_csv(p.output_file, sep=r'\s+', decimal='.')

    C = data['C'].to_numpy(float)
    R = np.maximum(data['R'].to_numpy(float), np.finfo(float).eps)
    G = np.maximum(data['G'].to_numpy(float), np.finfo(float).eps)
    B = np.maximum(data['B'].to_numpy(float), np.finfo(float).eps)

    expr_list = []
    label_list = []
    with open('index_list.txt', 'r') as f:
        for line in f:
            if '|' in line:
                expr, label = line.strip().split('|', 1)
                expr_list.append(expr.strip())
                label_list.append(label.strip())

    indices = []
    for expr in expr_list:
        try:
            idx_val = eval(expr, {'R': R, 'G': G, 'B': B, 'np': np})
        except Exception as e:
            print(f"Error evaluating {expr}: {e}")
            idx_val = np.full_like(R, np.nan)
        indices.append(idx_val)

    r_vals, R2_vals, RMSE_vals, MAPE_vals, red_thresholds = [], [], [], [], []

    for i, x in enumerate(indices):
        valid = ~np.isnan(x) & ~np.isnan(C)
        if np.sum(valid) < 3:
            r_vals.append(np.nan)
            R2_vals.append(np.nan)
            RMSE_vals.append(np.nan)
            MAPE_vals.append(np.nan)
            red_thresholds.append(np.nan)
            continue

        x_valid = x[valid]
        y_valid = C[valid]

        r = np.corrcoef(x_valid, y_valid)[0, 1]

        poly = np.polyfit(x_valid, y_valid, 1)
        y_fit = np.polyval(poly, x_valid)

        SS_res = np.sum((y_valid - y_fit) ** 2)
        SS_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        R2 = 1 - SS_res / SS_tot
        RMSE = np.sqrt(np.mean((y_valid - y_fit) ** 2))
        MAPE = np.mean(np.abs((y_valid - y_fit) / y_valid)) * 100

        target_conc = p.thr_fraction * np.max(y_valid)
        x_lin = np.linspace(np.min(x_valid), np.max(x_valid), 1000)
        y_lin = np.polyval(poly, x_lin)
        idx_t = np.argmin(np.abs(y_lin - target_conc))
        red_threshold = x_lin[idx_t]

        r_vals.append(r)
        R2_vals.append(R2)
        RMSE_vals.append(RMSE)
        MAPE_vals.append(MAPE)
        red_thresholds.append(red_threshold)

    T = pd.DataFrame({
        'Index': label_list,
        'r_Pearson': r_vals,
        'R2': R2_vals,
        'RMSE': RMSE_vals,
        'MAPE': MAPE_vals,
        'red_threshold': red_thresholds
    }).sort_values(by='R2', ascending=False)

    print("\n=== correlation results with fluorimeter ===")
    print(T)

    T.to_csv('results_indices.txt', sep='\t', index=False)
    print("\nSaved results to: out_indices.txt")

    best_row = T.iloc[0]
    best_label = best_row['Index']
    best_label = best_row['Index']
    expr = expr_list[label_list.index(best_label)]
    func_code = f"lambda R, G, B: {expr}"
    best_idx = indices[label_list.index(best_label)]

    valid = ~np.isnan(best_idx) & ~np.isnan(C)
    x = best_idx[valid]
    y = C[valid]

    poly = np.polyfit(x, y, 1)
    y_fit = np.polyval(poly, x)

    plt.figure(figsize=(6, 5))
    plt.plot(x, y, 'ok', markersize=4, label='Data')
    plt.plot(np.sort(x), np.polyval(poly, np.sort(x)), 'k-', linewidth=2, label='Linear fit')
    plt.xlabel(best_label)
    plt.ylabel('Rhodamine WT Concentration (ppb)')
    plt.title(f'Best correlation: {best_label}', fontweight='bold')
    plt.legend(title=f"R² = {best_row['R2']:.2f}, RMSE = {best_row['RMSE']:.2f}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best_model = {
        'Index': best_label,
        'Poly': poly.tolist(),          #
        'Func': func_code,
        'R2': float(best_row['R2']),
        'RMSE': float(best_row['RMSE']),
        'MAPE': float(best_row['MAPE']),
        'red_threshold': float(best_row['red_threshold'])
    }

    # Guarda el model en JSON (robust, net)
    with open("best_model.json", "w") as f:
        json.dump(best_model, f, indent=4)

    return best_model

