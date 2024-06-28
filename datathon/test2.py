import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import csv

# Load data
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
for ima in images[1:200]:
    for ou in outfits[1:100]:
        if ou[1] == ima[0]:
            seq = ["START"]
            for ou2 in outfits[1:100]:
                if ou[0] == ou2[0]:
                    seq.append(ou2[-1])
            seq.append("END")
            if len(seq) > maxlen:
                maxlen = len(seq)
            combined.append([ima[-1], seq])

# Save combined data to CSV
with open('datathon/datathon/dataset/combined_data.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(combined)

# Tokenize and pad sequences
sequence_data = [a for _, a in combined]
tokenizer = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts([str(seq) for seq in sequence_data])
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
        y_seq.append(sequ[s+1])

# Tokenize and pad input sequences
tokenized_sequences = tokenizer.texts_to_sequences([" ".join(seq) for seq in X_seq])
x_padded = pad_sequences(tokenized_sequences, maxlen=maxlen, padding='post', truncating='post')

# Tokenize and pad output sequences
tokenized_sequences = tokenizer.texts_to_sequences([" ".join(seq) for seq in y_seq])
y_padded = pad_sequences(tokenized_sequences, maxlen=maxlen, padding='post', truncating='post')

# Define the image branch
image_input = keras.Input(shape=(None, None, 3))  # Adjust input shape based on your images
encoded_image = layers.Conv2D(64, (3, 3), activation='relu')(image_input)
encoded_image = layers.Flatten()(encoded_image)

# Define the sequence branch
sequence_input = keras.Input(shape=(maxlen,))
embedding_dim = 50
embedded_sequence = layers.Embedding(num_classes, embedding_dim, input_length=maxlen)(sequence_input)
lstm_output = layers.LSTM(256)(embedded_sequence)

# Combine the branches
merged = layers.concatenate([encoded_image, lstm_output])

# Add more dense layers for final prediction
output = layers.Dense(num_classes, activation='softmax')(merged)

# Create the model
model = keras.Model(inputs=[image_input, sequence_input], outputs=output)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(
    x=[np.asarray(X_images), x_padded],
    y=y_padded,
    epochs=2,
    batch_size=32,
    validation_split=0.2
)

# Save the model
model.save('./model.keras')
