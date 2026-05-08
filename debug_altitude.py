import pandas as pd
import numpy as np
from droneVideoParser import DroneVideoCSVParser

parser = DroneVideoCSVParser('dataset/drone_footage/23-02-01_FR_F01_V01.mp4', 'dataset/drone_footage/23-02-01_FR_F01.csv')

# Перевіримо висоти
print('Перші 5 поз у ENU:')
for i in range(5):
    img, pose = parser[i]
    print(f'  Кадр {i}: pos={pose[:3,3]}, Z={pose[2,3]:.2f}')

print(f'\nБуферна висота з CSV (pos_csv):')
print(f'  pos_csv[0:5] = \n{parser.pos_csv[0:5]}')

print(f'\nКолони у CSV з "height" або "altitude":')
for col in parser.df.columns:
    if 'height' in col.lower() or 'altitude' in col.lower():
        print(f'  {col}: {parser.df[col].iloc[0:3].values}')

print(f'\nZ позиції від екстраполяції за часом:')
print(f'  times[0:5] = {parser.csv_times[0:5]}')
