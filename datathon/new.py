import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import csv
import cv2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the images data
images = []
with open('datathon/datathon/dataset/product_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        images.append(row)

numclasses = len(images) + 2
outfits = []
with open('datathon/datathon/dataset/outfit_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        outfits.append(row)

combined = []
maxlen = 0

for ima in images[1:1000]:
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

# Define the CNN for image processing
image = cv2.imread('datathon/' + combined[0][0].replace('"', ""))
height, width, channels = image.shape
image_input = keras.Input(shape=(height, width, channels))
encoded_image = layers.Conv2D(64, (3, 3), activation='relu')(image_input)
encoded_image = layers.Flatten()(encoded_image)

# Define the LSTM for sequence generation
embedding_dim = 50  # Adjust as needed
sequence_input = keras.Input(shape=(maxlen,))

tokenizer = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')

tokenizer.fit_on_texts([" ".join(seq) for _, seq in combined])
tokenized_sequence = tokenizer.texts_to_sequences([" ".join(combined[0][1])])
padded_sequence = pad_sequences(tokenized_sequence, maxlen=maxlen, padding='post', truncating='post')
embedded_sequence = layers.Embedding(numclasses, embedding_dim, input_length=maxlen)(sequence_input)
lstm_output = layers.LSTM(256)(embedded_sequence)

# Combine the encoded image and LSTM output
merged = layers.concatenate([encoded_image, lstm_output])

# Add more dense layers for final prediction
num_classes = 10  # Adjust as needed
output = layers.Dense(num_classes, activation='softmax')(merged)

# Create the model
model = keras.Model(inputs=[image_input, sequence_input], outputs=output)

# Assuming your classes are one-hot encoded
prediction = model.predict([np.expand_dims(image, axis=0), np.array(padded_sequence)])

predicted_class = np.argmax(prediction)

word_index = tokenizer.word_index
print(word_index)
index_to_word = {index: word for word, index in word_index.items()}
predicted_token = index_to_word[predicted_class]

print("Predicted Token:", predicted_token)