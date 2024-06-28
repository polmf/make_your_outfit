import numpy as np
#from fashion_clip.fashion_clip import FashionCLIP
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ORIGIN_INDEX =  59
images = []
dades = []
with open('C:/Users/Usuario/OneDrive/Escritorio/Datathon/datathon/dataset/dades_processades.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        element = row[-1]
        e = 'C:/Users/USER/Desktop/Datathon/' + element.replace('"', "")
        images.append(str(e))
        dades.append(row)

outfit_data = []
with open('C:/Users/Usuario/OneDrive/Escritorio/Datathon/datathon/dataset/outfit_prep_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        outfit_data.append(row)

images = images[1:]
dades = dades[1:]
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load your image embeddings
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

#Outfit mean embedings:
outfit_embeddings = {}
codis = [m[0] for m in metadata]
# Calculate the average embedding for each outfit
for outfit in outfit_data:
    outfit_id, item_id = outfit
    item_idx = codis.index(item_id)
    item_embedding = image_embeddings[item_idx]

    if outfit_id not in outfit_embeddings:
        outfit_embeddings[outfit_id] = item_embedding
    else:
        outfit_embeddings[outfit_id] += item_embedding

# Normalize outfit embeddings
for outfit_id, embedding in outfit_embeddings.items():
    outfit_embeddings[outfit_id] /= len(embedding)


metadata = [m[:9] for m in metadata]
categorical_columns = [0,1,2,3,4,5,6,7,8]

# Extract the values from the categorical columns
categorical_values = [[entry[col] for col in categorical_columns] for entry in metadata]
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(categorical_values)

metadata= onehot_encoded
print(metadata[0])
# Function to calculate similarity between images based on metadata
def calculate_similarity_based_on_metadata(embedding1, embedding2, metadata1, metadata2, metaoutfits1, metaoutfits2):
    # For simplicity, use cosine similarity
    embedding_similarity = np.dot(embedding1, embedding2)
    # Calculate cosine similarity between metadata
    metadata_similarity = cosine_similarity([metadata1], [metadata2])[0][0]
    
    meta1 = None
    for m1 in metaoutfits1:
        if meta1 == None:
            meta1 = outfit_embeddings[m1]
        else:
            meta1 += outfit_embeddings[m1]
    if meta1 != None:
        meta1 /= len(metaoutfits1)
    meta2 = None
    for m2 in metaoutfits2:
        if meta2== None:
            meta2 = outfit_embeddings[m2]
        else:
            meta2 += outfit_embeddings[m2]
    if meta2 != None:
        meta2 /= len(metaoutfits2)
    if meta1 != None and meta2 != None:
        outfits_similarity = np.dot(meta1, meta2)
    else:
        outfits_similarity = float("inf")
    # Combine similarities (you can adjust the weights based on importance)
    combined_similarity = 0.1 * embedding_similarity + 0.1* metadata_similarity + 0.8* outfits_similarity if outfits_similarity != float("inf") else 0.8 * embedding_similarity + 0.2* metadata_similarity
    return combined_similarity

# Example: Calculate similarity between the first two images based on metadata


similarities = []

for e in range(len(image_embeddings)):
    metadata_similarity_score = calculate_similarity_based_on_metadata(
    image_embeddings[ORIGIN_INDEX],
    image_embeddings[e],
    metadata[ORIGIN_INDEX],
    metadata[e],
    meta_outfits[ORIGIN_INDEX],
    meta_outfits[e]
)
    similarities.append(metadata_similarity_score)

outfit = [ORIGIN_INDEX]
min_dist = np.argsort(similarities)

Tipus_roba = []

def outfit_complet(outfit):
    return len(Tipus_roba) == 5

def check_append(outfit, m):
    accessories = 1
    i = True
    if dades[m][8] in Tipus_roba:
        return False
    for o in outfit:
        if dades[o][8] == dades[m][8]:
            i = False
    return i

for o in outfit:
    Tipus_roba.append(dades[o][8])
    if dades[o][8] == '"Dresses':
        Tipus_roba.append('Tops')
        Tipus_roba.append('Bottoms')
    
    elif dades[o][8] == 'Tops' or dades[o][8] == 'Bottoms':
        Tipus_roba.append('"Dresses')

i = 1
while outfit_complet(outfit) == False and i < len(similarities):
    m = min_dist[-i]
    if check_append(outfit, m):
        outfit.append(m)
        Tipus_roba.append(dades[m][8])
        if dades[o][8] == '"Dresses':
            if 'Tops' not in Tipus_roba and 'Bottoms' not in Tipus_roba:    
                Tipus_roba.append('Tops', 'Bottoms')
        
        elif dades[o][8] == 'Tops' or dades[o][8] == 'Bottoms':
            if '"Dresses' not in Tipus_roba:   
                Tipus_roba.append('"Dresses')

    i += 1

for e in outfit:
    print(dades[e][8])

outfit_images= [] 
for e in outfit:
     outfit_images.append( images[e])

outfit_complerts = []
for e in outfit:
    outfit_complerts.append( dades[e])

print(outfit)
gs = gridspec.GridSpec(3, 4, wspace=0.1, hspace=0.2)

print(outfit_images)
print(outfit_complerts)
# Loop through the images and plot them
for i, image_path in enumerate(outfit_images):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    ax = plt.subplot(gs[i])
    ax.imshow(img)
    ax.set_title(f'Image {i + 1}')
    ax.axis('off')

plt.show()
