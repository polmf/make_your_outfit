import numpy as np
import csv
import pickle



images = []
dades = []
with open('datathon/datathon/dataset/dades_processades.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        element = row[-1]
        e = 'C:/Users/Usuari/OneDrive/Documentos/Datathon/aguacate/datathon/' + element.replace('"', "")
        images.append(str(e))
        dades.append(row)

outfit_data = []
with open('datathon/datathon/dataset/outfit_prep_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        outfit_data.append(row)

images = images[1:]
dades = dades[1:]




image_embeddings = np.load('image_embeddings_prep.npy')


outfit_data = outfit_data[1:]
metadata = dades
combined = []

for ima in dades:
    aa = []
    for ou in outfit_data:
        if ou[1] == ima[0]:
            aa.append(ou[0])
    combined.append(aa)
meta_outfits = combined

with open('meta_outfits.csv', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
    for meta in meta_outfits:
        write.writerow(meta)

#Outfit mean embedings:
outfit_embeddings = {}
outfit_counts = {}
codis = [m[0] for m in metadata]
# Calculate the average embedding for each outfit
for outfit in outfit_data:
    outfit_id, item_id = outfit
    item_idx = codis.index(item_id)
    item_embedding = image_embeddings[item_idx]

    if outfit_id not in outfit_embeddings:
        outfit_embeddings[outfit_id] = item_embedding
        outfit_counts[outfit_id] = 1
    else:
        outfit_embeddings[outfit_id] += item_embedding
        outfit_counts[outfit_id] += 1
# Normalize outfit embeddings
for outfit_id, embedding in outfit_embeddings.items():
    outfit_embeddings[outfit_id] /= outfit_counts[outfit_id]

with open('outfit_embeddings.pkl', 'wb') as fp:
    pickle.dump(outfit_embeddings, fp)
    print('dictionary saved successfully to file')

