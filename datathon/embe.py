
from fashion_clip.fashion_clip import FashionCLIP
import csv
import numpy as np

fclip = FashionCLIP('fashion-clip')
images = []
with open('datathon/datathon/dataset/dades_processades.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        element = row[-1]
        e = 'C:/Users/Usuari/OneDrive/Documentos/Datathon/aguacate/datathon/' + element.replace('"', "")
        images.append(str(e))

print(images[0])

print(images)
# we create image embeddings and text embeddings
image_embeddings = fclip.encode_images(images[1:], batch_size=32)

# we normalize the embeddings to unit norm (so that we can use dot product instead of cosine similarity to do comparisons)
image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
np.save('image_embeddings_prep.npy', image_embeddings)