import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


breed_labels = {
    0: "cairn",
    1: "border collie",
    }  # Mapping of breed name to label


def test(img_dir, net, image_size=(224, 224)):
    '''
    show the model's prediction for all jpg images in img_dir
    '''
    for image_name in os.listdir(img_dir):
        if image_name.endswith(".jpg"):
            # get prediction
            image_path = os.path.join(img_dir, image_name)

            img = Image.open(image_path)

            img = img.resize(image_size)

            img_array = np.array(img) / 255.0

            img_array_flattened = img_array.flatten()

            pred = np.argmax(net.feedforward(img_array_flattened))

            breed_label = breed_labels[pred]

            # show image
            img = mpimg.imread(image_path)

            fig, ax = plt.subplots()

            ax.imshow(img)
            ax.text(0, 50, breed_label, color='black', fontsize=12, ha='center', va='center')

            ax.axis('off')

            plt.show()
            
