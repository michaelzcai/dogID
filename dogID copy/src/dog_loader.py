import os
import numpy as np
from PIL import Image

def load_dogs_data(image_size=(224, 224), train_size=0.8, val_size=0.1):

    images = []
    labels = []
    breed_labels = {}  # Mapping of breed name to label
    
    # Get a sorted list of subdirectories (dog breeds)
    breed_names = sorted(os.listdir('../shortdogimages'))
    print(breed_names)
    
    for label, breed_name in enumerate(breed_names):
        breed_folder = os.path.join('../shortdogimages', breed_name)
        
        # Check if it's a directory (for images)
        if os.path.isdir(breed_folder):
            for image_name in os.listdir(breed_folder):
                if image_name.endswith(".jpg"):
                    image_path = os.path.join(breed_folder, image_name)
                    
                    # Open the image using Pillow
                    img = Image.open(image_path)
                    
                    # Resize the image to the target size
                    img = img.resize(image_size)
                    
                    # Convert image to numpy array and normalize pixel values
                    img_array = np.array(img) / 255.0
                    
                    # Flatten the image array to 1D
                    img_array_flattened = img_array.flatten()
                    
                    # Append the image data and label
                    images.append(img_array_flattened)
                    labels.append(label)
    
    # Convert to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)
    vectorized_labels = [vectorized_result(y) for y in labels]
    vectorized_labels = np.array(vectorized_labels)
    
    # shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    vectorized_labels = vectorized_labels[indices]
    
    # Compute split sizes
    num_train = int(len(images) * train_size)
    num_val = int(len(images) * val_size)
    
    X_train = images[:num_train]
    y_train = vectorized_labels[:num_train]
    
    X_val = images[num_train:num_train+num_val]
    y_val = labels[num_train:num_train+num_val]
    
    X_test = images[num_train+num_val:]
    y_test = labels[num_train+num_val:]

    #get data
    training_data = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    validation_data = [(X_val[i], y_val[i]) for i in range(len(X_val))]
    test_data = [(X_test[i], y_test[i]) for i in range(len(X_test))]
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """return a 2-dimensional unit vector with a 1.0 in the jth position and
    zeros elsewhere
    """
    e = np.zeros((2,1))
    e[j] = 1.0
    return e


