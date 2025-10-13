"""
Created on Fri Oct 10 2025 ‏‎‏‎17:59:38
Code part of RHOD (UPC-BarcelonaTech)
Last modified on Mon Oct 13 2025
@author: raquel peñas torramilans
@contact: raquel.penas@upc.edu
"""

import numpy as np
import pandas as pd
from scipy.io import savemat
import matplotlib.pyplot as plt
from pathlib import Path

data = pd.read_csv('insitu_rho.txt', sep=r'\s+', decimal='.')

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

    p = np.polyfit(x_valid, y_valid, 1)
    y_fit = np.polyval(p, x_valid)

    SS_res = np.sum((y_valid - y_fit) ** 2)
    SS_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    R2 = 1 - SS_res / SS_tot
    RMSE = np.sqrt(np.mean((y_valid - y_fit) ** 2))
    MAPE = np.mean(np.abs((y_valid - y_fit) / y_valid)) * 100

    target_conc = 0.05 * np.max(y_valid)
    x_lin = np.linspace(np.min(x_valid), np.max(x_valid), 1000)
    y_lin = np.polyval(p, x_lin)
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
print("\nSaved results to: results_indices.txt")

best_row = T.iloc[0]
best_label = best_row['Index']
best_idx = indices[label_list.index(best_label)]

valid = ~np.isnan(best_idx) & ~np.isnan(C)
x = best_idx[valid]
y = C[valid]

p = np.polyfit(x, y, 1)
y_fit = np.polyval(p, x)

plt.figure(figsize=(6, 5))
plt.plot(x, y, 'ok', markersize=4, label='Data')
plt.plot(np.sort(x), np.polyval(p, np.sort(x)), 'k-', linewidth=2, label='Linear fit')
plt.xlabel(best_label)
plt.ylabel('Rhodamine WT Concentration (ppb)')
plt.title(f'Best correlation: {best_label}', fontweight='bold')
plt.legend(title=f"R² = {best_row['R2']:.2f}, RMSE = {best_row['RMSE']:.2f}")
plt.grid(True)
plt.tight_layout()
plt.show()

try:
    func_code = f"lambda R, G, B: {expr_list[label_list.index(best_label)]}"
    func_handle = eval(func_code, {'np': np})
except Exception as e:
    print(f"Could not create function handle for {best_label}: {e}")
    func_handle = None

model_concentration = {
    'best': {
        'Index': best_label,
        'Poly': p,
        'R2': best_row['R2'],
        'RMSE': best_row['RMSE'],
        'MAPE': best_row['MAPE'],
        'Func': func_code,
        'red_threshold': best_row['red_threshold']
    }
}

best_model = model_concentration['best']
print(f" - Best model: {best_label}")
