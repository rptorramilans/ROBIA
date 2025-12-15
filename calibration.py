"""
Created on Fri Nov 5 2025
Code part of ROBIA (UPC-BarcelonaTech)
Last modified on Fri Nov 5 2025
@author: Raquel Peñas Torramilans
@contact: raquel.penas@upc.edu
"""

import os
import pandas as pd
import numpy as np
import rasterio
from datetime import datetime

# preprocessing parameters
import params as p

# 1) llegir fitxer de coordenades
tabla = pd.read_csv(p.coords_file, sep=';', decimal='.')

# 2) normalitzar format de l’hora i llegir noms de les imatges i extreure hores disponibles
if np.issubdtype(tabla['HORA'].dtype, np.datetime64):
    tabla['HORA'] = tabla['HORA'].dt.strftime('%H:%M')
else:
    tabla['HORA'] = tabla['HORA'].astype(str).str.extract(r'(\d{1,2}:\d{2})')[0]

result_rows = []

files = [f for f in os.listdir(p.input_folder) if f.lower().endswith('.tiff')]
available_times = []
for f in files:
    try:
        t = f.split('.')[0]  # treu extensió
        t = t.replace('_', ':')  # "11_16" -> "11:16"
        datetime.strptime(t, "%H:%M")  # comprovar format
        available_times.append(t)
    except:
        pass

# 3) iterar per hora
for hora in sorted(tabla['HORA'].unique()):
    # buscar la imatge amb hora més propera
    try:
        hora_dt = datetime.strptime(hora, "%H:%M")
        diffs = [abs((hora_dt - datetime.strptime(ht, "%H:%M")).total_seconds()) for ht in available_times]
        nearest_time = available_times[np.argmin(diffs)]
        hora_file = nearest_time.replace(':', '_')
    except Exception:
        print(f"No valid image found for hour {hora}")
        continue

    img_path = os.path.join(p.input_folder, f"{hora_file}.tiff")

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    try:
        with rasterio.open(img_path) as src:
            img = src.read()  # shape = (bands, rows, cols)
            transform = src.transform
    except Exception as e:
        print(f"Error while reading {img_path}: {e}")
        continue

    subset = tabla[tabla['HORA'] == hora].copy()
    Rvals, Gvals, Bvals = [], [], []

    for x, y in zip(subset['X'], subset['Y']):
        try:
            col, row = ~transform * (x, y)
            col, row = int(round(col)), int(round(row))

            if 0 <= row < img.shape[1] and 0 <= col < img.shape[2]:
                Rvals.append(float(img[0, row, col]))
                Gvals.append(float(img[1, row, col]))
                Bvals.append(float(img[2, row, col]))
            else:
                Rvals += [np.nan]
                Gvals += [np.nan]
                Bvals += [np.nan]
        except Exception:
            Rvals += [np.nan]
            Gvals += [np.nan]
            Bvals += [np.nan]

    subset['R'] = Rvals
    subset['G'] = Gvals
    subset['B'] = Bvals
    subset['IMATGE_USADA'] = hora_file  # opcional per saber quina imatge s'ha fet servir

    result_rows.append(subset)

# 4) combinar resultats
if result_rows:
    tabla_resultado = pd.concat(result_rows, ignore_index=True)
else:
    print("RGB not extracted, no results generated.")
    exit()

# 5) guardar resultats
if p.output_file.lower().endswith(('.xlsx', '.xls')):
    tabla_resultado.to_excel(p.output_file, index=False)
else:
    tabla_resultado.to_csv(p.output_file, index=False, sep='\t')

print(f"File saved correctly: {p.output_file}")


import numpy as np
import pandas as pd
from scipy.io import savemat
import matplotlib.pyplot as plt
from pathlib import Path

import params as p

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

try:
    func_code = f"lambda R, G, B: {expr_list[label_list.index(best_label)]}"
    func_handle = eval(func_code, {'np': np})
except Exception as e:
    print(f"Could not create function handle for {best_label}: {e}")
    func_handle = None

model_concentration = {
    'best': {
        'Index': best_label,
        'Poly': poly,
        'R2': best_row['R2'],
        'RMSE': best_row['RMSE'],
        'MAPE': best_row['MAPE'],
        'Func': func_code,
        'red_threshold': best_row['red_threshold']
    }
}

best_model = model_concentration['best']
