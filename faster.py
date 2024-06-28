import sys
import numpy as np
import pygame
import csv
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import random
import pyperclip

ORIGIN_INDEX =  10
images = []
dades = []
with open('C:/Users/USER/Desktop/Datathon/datathon/dataset/dades_reprocessades_bones.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        element = row[-1]
        e = 'C:/Users/USER/Desktop/Datathon/' + element.replace('"', "")
        images.append(str(e))
        dades.append(row)

outfit_data = []
with open('C:/Users/USER/Desktop/Datathon/datathon/dataset/outfit_prep_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        outfit_data.append(row)

import pickle

with open('C:/Users/USER/Desktop/Datathon/aguacate/outfit_embeddings.pkl', 'rb') as fp:
    outfit_embeddings = pickle.load(fp)

images = images[1:]
dades = dades[1:]

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

pygame.init()
width, height = 1200, 800  # Increase screen dimensions
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Outfit Visualizer")

# Set up fonts
font_large = pygame.font.Font(None, 48)  # Increase font size for main display
font_small = pygame.font.Font(None, 36)  # Font size for selection screen

# Function to display outfit images
def display_outfit(outfit_images):
    screen.fill((255, 255, 255))

    total_width = (width - 100)  # Total available width for the images
    image_width = total_width / len(outfit_images)

    # Calculate the starting position to center the images
    start_position = (width - total_width) / 2

    for i, image_path in enumerate(outfit_images):
        img = pygame.image.load(image_path)
        img = pygame.transform.scale(img, (int(image_width), int(image_width)))  # Cast to int for position
        screen.blit(img, (start_position + i * image_width, 300))  # Adjust spacing and positioning

    pygame.display.flip()

def display_loading_screen():
    screen.fill((255, 255, 255))

    loading_text = font_large.render("Loading...", True, (0, 0, 0))
    screen.blit(loading_text, (width // 2 - loading_text.get_width() // 2, height // 2))

    pygame.display.flip()

# Function to display selection screen
def display_selection_screen(images):
    screen.fill((255, 255, 255))

    title_text = font_large.render("Choose Your First Clothing Piece", True, (0, 0, 0))
    screen.blit(title_text, (width // 2 - title_text.get_width() // 2, 100))

    instructions_text = font_small.render("Click on an item to select", True, (0, 0, 0))
    screen.blit(instructions_text, (width // 2 - instructions_text.get_width() // 2, 200))

    for i, image_path in enumerate(images):
        img = pygame.image.load(image_path)
        img = pygame.transform.scale(img, (150, 150))
        screen.blit(img, (i * 170, 300))

    pygame.display.flip()

step = random.randint(1, len(images)-8)
# Get the initial outfit selection from the user
initial_images = images[step:step+7]  # Display the first 5 images on the selection screen
display_selection_screen(initial_images)


selected_item = None
while selected_item is None:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            clicked_index = x // 170
            if 0 <= clicked_index < len(initial_images):
                selected_item = initial_images[clicked_index]
                display_loading_screen()
                ORIGIN_INDEX = clicked_index + step
                pygame.display.flip()
                print("Loading...")
                break

print(selected_item)


# Load your image embeddings
image_embeddings = np.load('C:/Users/USER/Desktop/Datathon/aguacate/image_embeddings_prep.npy')

print("hola")
outfit_data = outfit_data[1:]
metadata = dades
metadata = [m[:9] for m in metadata]
categorical_columns = [0,1,2,3,4,5,6,7,8]

# Extract the values from the categorical columns
categorical_values = [[entry[col] for col in categorical_columns] for entry in metadata]
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(categorical_values)

metadata= onehot_encoded
# Function to calculate similarity between images based on metadata
def calculate_similarity_based_on_metadata(embedding1, embedding2, metadata1, metadata2, metaoutfits1, metaoutfits2):
    embedding_similarity = np.dot(embedding1, embedding2)
    metadata_similarity = cosine_similarity([metadata1], [metadata2])[0][0]

    meta1 = [None]
    for m1 in metaoutfits1:
        if meta1[0] == None:
            meta1 = outfit_embeddings[m1]
        else:
            meta1 += outfit_embeddings[m1]
    if meta1[0] != None:
        meta1 /= len(metaoutfits1)
    meta2 = [None]
    for m2 in metaoutfits2:
        if meta2[0]== None:
            meta2 = outfit_embeddings[m2]
        else:
            meta2 += outfit_embeddings[m2]
    if meta2[0] != None:
        meta2 /= len(metaoutfits2)
    if meta1[0] != None and meta2[0] != None:
        outfits_similarity = np.dot(meta1, meta2)
    else:
        outfits_similarity = 0
    combined_similarity = 0.5 * embedding_similarity + 0.3*metadata_similarity +0.5*outfits_similarity
    return combined_similarity

similarities = []
outfit = [ORIGIN_INDEX]

meta_outfits = []
with open('C:/Users/USER/Desktop/Datathon/aguacate/meta_outfits.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        meta_outfits.append(row)


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



min_dist = np.argsort(similarities)


list_removed = []
Tipus_roba = set()

def outfit_complet(outfit):
    return len(Tipus_roba) >= 6

def check_append(outfit, m, list_removed):
    accessories = 1
    if m in list_removed:
        return False
    i = True
    if dades[m][8] in Tipus_roba and dades[m][8] != 'Accesories, Swim and Intimate':
        return False
    elif dades[m][8] == 'Accesories, Swim and Intimate' and  dades[m][11] == 'Shoes' and 'Shoes' not in Tipus_roba:
        return True
    for o in outfit:
        if dades[o][8] == dades[m][8]:
            if dades[o][8] == 'Accesories, Swim and Intimate':
                if dades[o][11] == 'Shoes' and  dades[o][11] == dades[m][11]:
                    i = False
                elif dades[o][11] != 'Shoes':
                    i = False
            else:
                i = False
    return i

for o in outfit:
    if dades[o][8] == 'Accesories, Swim and Intimate':
        if dades[o][11] == 'Shoes':
            Tipus_roba.add('Shoes')
        elif dades[o][8] =='Accesories, Swim and Intimate':
            Tipus_roba.add('Accesories, Swim and Intimate')
    else:
        Tipus_roba.add(dades[o][8])
    if dades[o][8] == 'Dresses, jumpsuits and Complete set':
        Tipus_roba.add('Tops')
        Tipus_roba.add('Bottoms')
    
    elif dades[o][8] == 'Tops' or dades[o][8] == 'Bottoms':
        Tipus_roba.add('Dresses, jumpsuits and Complete set')

i = 1
while outfit_complet(outfit) == False and i < len(similarities):
    m = min_dist[-i]
    if check_append(outfit, m, list_removed):
        outfit.append(m)
        
        if dades[m][8] == 'Dresses, jumpsuits and Complete set':
            Tipus_roba.add('Tops')
            Tipus_roba.add('Bottoms')
        
        elif dades[m][8] == 'Tops' or dades[m][8] == 'Bottoms':
            Tipus_roba.add('Dresses, jumpsuits and Complete set')
        
        if dades[m][8] == 'Accesories, Swim and Intimate':
            if dades[m][11] == 'Shoes':
                Tipus_roba.add('Shoes')
            elif dades[m][8] =='Accesories, Swim and Intimate':
                Tipus_roba.add('Accesories, Swim and Intimate')
        else:
            Tipus_roba.add(dades[m][8])

    i += 1



desired_order = ['Dresses, jumpsuits and Complete set', 'Outerwear', 'Tops', 'Bottoms' ,'Accesories, Swim and Intimate']
outfit.sort(key=lambda x: desired_order.index(dades[x][8]) if dades[x][11] != 'Shoes' else float("inf"))

outfit_images= [] 
for e in outfit:
     outfit_images.append( images[e])

outfit_complerts = []
for e in outfit:
    outfit_complerts.append( dades[e])

o_sim = []

for e in outfit:
        o_sim.append(similarities[e])
print(o_sim)


print(outfit_complerts)

gs = gridspec.GridSpec(3, 4, wspace=0.1, hspace=0.2)

# Loop through the images and plot them
for i, image_path in enumerate(outfit_images):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    ax = plt.subplot(gs[i])
    ax.imshow(img)
    ax.set_title(f'Image {i + 1}')
    ax.axis('off')

old_i = 1
min_dist = np.argsort(similarities)
# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos

            total_width = (width - 100)  # Total available width for the images
            image_width = total_width / len(outfit_images)

            # Calculate the starting position to center the images
            total_width = (width - 100)
            image_width = total_width / len(outfit_images)
            start_position = (width - total_width) / 2

            clicked_index = int((x - start_position) / image_width)  # Adjust based on new spacing
            if 0 <= clicked_index < len(outfit):
                removed_item = outfit.pop(clicked_index)
                list_removed.append(removed_item)
                if dades[removed_item][8] == 'Accesories, Swim and Intimate' and  dades[removed_item][11] == 'Shoes':
                    Tipus_roba.remove('Shoes')
                else:
                    Tipus_roba.remove(dades[removed_item][8])
                print(f"Removed: {dades[removed_item][8]}")
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
            a = " ".join([str(dades[ou][0]) for ou in outfit])
            pyperclip.copy(a)
            spam = pyperclip.paste()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
            # Reset to the initial screen
            step = random.randint(1, len(images) - 8)
            initial_images = images[step:step + 7]
            selected_item = None
            display_selection_screen(initial_images)
            outfit = [step + clicked_index]  # Assuming clicked_index is still valid
            list_removed = []
            Tipus_roba = set()
            selected_item = None
            while selected_item is None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        clicked_index = x // 170
                        if 0 <= clicked_index < len(initial_images):
                            selected_item = initial_images[clicked_index]
                            display_loading_screen()
                            ORIGIN_INDEX = clicked_index + step
                            pygame.display.flip()
                            print("Loading...")
                            break

            print(selected_item)


            # Load your image embeddings
            image_embeddings = np.load('C:/Users/USER/Desktop/Datathon/aguacate/image_embeddings_prep.npy')

            print("hola")
            outfit_data = outfit_data[1:]
            metadata = dades
            metadata = [m[:9] for m in metadata]
            categorical_columns = [0,1,2,3,4,5,6,7,8]

            # Extract the values from the categorical columns
            categorical_values = [[entry[col] for col in categorical_columns] for entry in metadata]
            onehot_encoder = OneHotEncoder(sparse=False)
            onehot_encoded = onehot_encoder.fit_transform(categorical_values)

            metadata= onehot_encoded
            
            similarities = []
            outfit = [ORIGIN_INDEX]

            meta_outfits = []
            with open('C:/Users/USER/Desktop/Datathon/aguacate/meta_outfits.csv', newline='') as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    meta_outfits.append(row)


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



            min_dist = np.argsort(similarities)


            list_removed = []
            Tipus_roba = set()

            for o in outfit:
                if dades[o][8] == 'Accesories, Swim and Intimate':
                    if dades[o][11] == 'Shoes':
                        Tipus_roba.add('Shoes')
                    elif dades[o][8] =='Accesories, Swim and Intimate':
                        Tipus_roba.add('Accesories, Swim and Intimate')
                else:
                    Tipus_roba.add(dades[o][8])
                if dades[o][8] == 'Dresses, jumpsuits and Complete set':
                    Tipus_roba.add('Tops')
                    Tipus_roba.add('Bottoms')
                
                elif dades[o][8] == 'Tops' or dades[o][8] == 'Bottoms':
                    Tipus_roba.add('Dresses, jumpsuits and Complete set')

            i = 1
            while outfit_complet(outfit) == False and i < len(similarities):
                m = min_dist[-i]
                if check_append(outfit, m, list_removed):
                    outfit.append(m)
                    
                    if dades[m][8] == 'Dresses, jumpsuits and Complete set':
                        Tipus_roba.add('Tops')
                        Tipus_roba.add('Bottoms')
                    
                    elif dades[m][8] == 'Tops' or dades[m][8] == 'Bottoms':
                        Tipus_roba.add('Dresses, jumpsuits and Complete set')
                    
                    if dades[m][8] == 'Accesories, Swim and Intimate':
                        if dades[m][11] == 'Shoes':
                            Tipus_roba.add('Shoes')
                        elif dades[m][8] =='Accesories, Swim and Intimate':
                            Tipus_roba.add('Accesories, Swim and Intimate')
                    else:
                        Tipus_roba.add(dades[m][8])

                i += 1


            desired_order = ['Dresses, jumpsuits and Complete set', 'Outerwear', 'Tops', 'Bottoms' ,'Accesories, Swim and Intimate']
            outfit.sort(key=lambda x: desired_order.index(dades[x][8]) if dades[x][11] != 'Shoes' else float("inf"))

            outfit_images= [] 
            for e in outfit:
                outfit_images.append( images[e])

            outfit_complerts = []
            for e in outfit:
                outfit_complerts.append( dades[e])


            print(outfit_complerts)

            gs = gridspec.GridSpec(3, 4, wspace=0.1, hspace=0.2)

            # Loop through the images and plot them
            for i, image_path in enumerate(outfit_images):
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                ax = plt.subplot(gs[i])
                ax.imshow(img)
                ax.set_title(f'Image {i + 1}')
                ax.axis('off')

            old_i = 1
            min_dist = np.argsort(similarities)
        i = 1
        while outfit_complet(outfit) == False and i < len(similarities):
            m = min_dist[-i]
            if check_append(outfit, m, list_removed):
                outfit.append(m)
                if dades[m][8] == 'Accesories, Swim and Intimate':
                    if dades[m][11] == 'Shoes':
                        Tipus_roba.add('Shoes')
                    elif dades[m][8] == 'Accesories, Swim and Intimate':
                        Tipus_roba.add('Accesories, Swim and Intimate')
                else:
                    Tipus_roba.add(dades[m][8])
                if dades[o][8] == 'Dresses, jumpsuits and Complete set':
                    Tipus_roba.add('Tops')
                    Tipus_roba.add('Bottoms')
                
                elif dades[o][8] == 'Tops' or dades[o][8] == 'Bottoms':
                    Tipus_roba.add('Dresses, jumpsuits and Complete set')
            i += 1
        old_i = i
    outfit.sort(key=lambda x: desired_order.index(dades[x][8]) if dades[x][11] != 'Shoes' else float("inf"))
    outfit_images = [images[e] for e in outfit]
    display_outfit(outfit_images)
    pygame.event.pump()
    pygame.time.delay(30)