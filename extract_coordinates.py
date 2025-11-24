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
