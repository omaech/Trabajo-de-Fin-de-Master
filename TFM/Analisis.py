import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo :
df = pd.read_csv('datos_sensor.csv', header = None)

# Ver las primeras filas para confirmar que se ha cargado correctamente
print(df.head())