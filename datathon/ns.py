import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

import cv2
import csv

# Load data using generators
# Define your own image loading function
def load_image(image_path):
    return cv2.imread(image_path.replace('"', ""))

# Image data generator with augmentation
image_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Combine data
# (Your data combining code remains the same)
images = []
with open('datathon/datathon/dataset/product_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        images.append(row)

outfits = []
with open('datathon/datathon/dataset/outfit_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        outfits.append(row)

# Combine data
combined = []
maxlen = 0
for ima in images[1:100]:
    for ou in outfits[1:200]:
        if ou[1] == ima[0]:
            seq = ["START"]
            for ou2 in outfits[1:200]:
                if ou[0] == ou2[0]:
                    seq.append(ou2[-1])
            if len(seq) > maxlen:
                maxlen = len(seq)
            combined.append([ima[-1], seq])

# Save combined data to CSV
with open('datathon/datathon/dataset/combined_data.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(combined)

# Tokenize and pad sequences
sequence_data = [[e.replace('"', "") for e in a] for _, a in combined]

z = []
for e in sequence_data:
    for l in e:
        z.append(l)
tokenizer = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts([str(seq) for seq in z])
num_classes = len(tokenizer.word_index) + 1

X_images = []
X_seq = []
y_seq = []

# Prepare data for training
for yes in range(len(combined)):
    sequ = sequence_data[yes]
    for s in range(1, len(sequ)-1):
        X_images.append(cv2.imread('datathon/' + combined[yes][0].replace('"', "")))
        X_seq.append(sequ[:s])
        y_seq.append(str(sequ[s+1]).replace('"', ''))

# Tokenize and pad input sequences
tokenized_sequences = tokenizer.texts_to_sequences([" ".join(seq) for seq in X_seq])
x_padded = pad_sequences(tokenized_sequences, maxlen=num_classes, padding='post', truncating='post')

# Tokenize and pad output sequences
tokenized_sequences_y = tokenizer.texts_to_sequences(y_seq)
# One-hot encode the tokenized output sequences
y_one_hot = to_categorical(tokenized_sequences_y, num_classes=num_classes)

# Tokenize and pad sequences
# (Your tokenization and padding code remains the same)

# Define the image branch
image_input = keras.Input(shape=(334, 239, 3))  # Adjust input shape based on your images
augmented_image = layers.Lambda(lambda x: image_datagen.random_transform(x))(image_input)  # Augmentation
augmented_image = layers.Conv2D(64, (3, 3), activation='relu')(augmented_image)
augmented_image = layers.Flatten()(augmented_image)

# (The rest of your code remains the same)

# Use a learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.9
    return lr

lr_schedule = LearningRateScheduler(lr_scheduler)

# Use early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Create the model
model = keras.Model(inputs=[image_input, sequence_input], outputs=output)

# Use Adam optimizer with a scheduler
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Train the model with generators
model.fit(
    x=[X_images, x_padded],
    y=y_one_hot,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping, lr_schedule]
)
