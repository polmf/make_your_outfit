import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input, Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Sample datasets
outfit_data = [
    (1, "51000622-02"),
    (1, "43067759-01"),
    # ... (remaining outfit data)
    (2086, "53015780-99")
]

clothing_data = [
    ("41085800-02", "02", "OFFWHITE", "WHITE", "Female", "Adult", "SHE", "P-PLANA", "Bottoms", "Trousers & leggings", "Trousers", "Trousers", "datathon/images/2019_41085800_02.jpg")
    # ... (remaining clothing data)
]

# Convert datasets to numpy arrays
outfit_data = np.array(outfit_data)
clothing_data = np.array(clothing_data)

# Extract features and labels
X_outfit = outfit_data[:, 1]
y_outfit = outfit_data[:, 0]

# Label encode categorical features in clothing_data
label_encoders = {}
for i in range(clothing_data.shape[1]):
    le = LabelEncoder()
    clothing_data[:, i] = le.fit_transform(clothing_data[:, i])
    label_encoders[i] = le

# Load and preprocess images
image_paths = clothing_data[:, -1]
images = [img_to_array(load_img(path, target_size=(224, 224))) for path in image_paths]
images = np.array(images)
images = preprocess_input(images)

# Create a mapping between clothing items and numerical indices
clothing_items = list(set(X_outfit))
clothing_to_index = {item: index for index, item in enumerate(clothing_items)}

# Convert the outfit dataset to numerical format
X_outfit_encoded = np.array([clothing_to_index[item] for item in X_outfit])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(X_outfit_encoded, y_outfit, images, test_size=0.2, random_state=42)

# Neural network model for text features
embedding_dim = 10

input_outfit = Input(shape=(1,))
embedding_outfit = Embedding(input_dim=len(clothing_items), output_dim=embedding_dim)(input_outfit)
flatten_outfit = Flatten()(embedding_outfit)

input_clothing = Input(shape=(clothing_data.shape[1],))
dense_clothing = Dense(32, activation='relu')(input_clothing)

# Neural network model for image features
input_image = Input(shape=(224, 224, 3))
conv1 = Conv2D(32, (3, 3), activation='relu')(input_image)
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D((2, 2))(conv2)
global_avg_pooling = GlobalAveragePooling2D()(pool2)

# Concatenate text and image features
concatenated = Concatenate()([flatten_outfit, dense_clothing, global_avg_pooling])
output = Dense(1, activation='linear')(concatenated)

model = Model(inputs=[input_outfit, input_clothing, input_image], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_train, clothing_data[:len(X_train)], img_train], y_train, epochs=10, batch_size=1)

# Evaluate the model
test_loss = model.evaluate([X_test, clothing_data[len(X_train):], img_test], y_test)
print(f"Test Loss: {test_loss}")

# Given a piece of clothing, predict the outfit
def predict_outfit(piece_of_clothing, image_path):
    piece_index = clothing_to_index[piece_of_clothing]
    clothing_input = clothing_data[0].reshape(1, -1)  # Assuming you are using the first entry in clothing_data for prediction
    image = img_to_array(load_img(image_path, target_size=(224, 224)))
    image = preprocess_input(np.array([image]))

    outfit_prediction = model.predict([np.array([piece_index]), clothing_input, image])
    suggested_outfit = np.round(outfit_prediction[0][0])
    
    return [index_to_clothing[i] for i in range(len(clothing_items)) if y_outfit[i] == suggested_outfit]

# Test the model with a piece of clothing and an image
piece_of_clothing_to_predict = "43063724-OR"
image_path_to_predict = "datathon/images/2019_41085800_02.jpg"
predicted_outfit = predict_outfit(piece_of_clothing_to_predict, image_path_to_predict)
print(f"Suggested outfit for {piece_of_clothing_to_predict}: {predicted_outfit}")
