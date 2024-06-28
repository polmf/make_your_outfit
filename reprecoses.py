import pandas as pd

# Carga de datos desde un CSV (Ejemplo)
data = pd.read_csv('C:/Users/USER/Desktop/Datathon/datathon/dataset/dades_processades.csv')

# Actualización del valor en la columna 'tu_columna'
data.loc[data['des_product_type'] == "Sandals", 'des_product_type'] = "Shoes"
data.loc[data['des_product_type'] == "Ankle Boots", 'des_product_type'] = "Shoes"
data.loc[data['des_product_type'] == "Boots", 'des_product_type'] = "Shoes"
data.loc[data['des_product_type'] == "Trainers", 'des_product_type'] = "Shoes"

print(data['des_product_type'].value_counts())

# Guardar el DataFrame modificado en un archivo CSV
data.to_csv('C:/Users/USER/Desktop/Datathon/datathon/dataset/dades_reprocessades_bones.csv', index=False)
