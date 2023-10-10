import os
from PIL import Image
import numpy as np


def load_data():
    # We are setting the path for our dataset of crops
    data_path = 'Agricultural-crops/'

    # here we r identifying group of crops
    crops = os.listdir(data_path)
    crops.sort()

    # Now creating empty list for images and labels
    images = []
    labels = []

    #making standard size for our image to avoid issues
    standard_size = (32, 32)

    # Here we r starting to load our dataset
    for crops_label in crops:
        crops_path = os.path.join(data_path, crops_label)
        for img_name in os.listdir(crops_path):
            #Here identifying for the path of images
            img_path = os.path.join(crops_path, img_name)
            img = Image.open(img_path)

            # Here how we studied made with rgb
            img = img.convert('RGB')
            # resizing by 32x32
            img = img.resize(standard_size)

            # Here we r using numpy to add image to arrays then flatten it
            img_array = np.array(img).flatten()

            # Now here we r checking for the errors (32 x 32 x 3 -> width x height x channel (RGB))
            if img_array.shape[0] != 32 * 32 * 3:
                print(f"Image {img_name} in class {crops_label} has an unexpected shape: {img_array.shape}")
                continue

            images.append(img_array)
            labels.append(crops_label)

    # Here now converting crops to integers
    new_labels = list(set(labels))
    integer_labels = [new_labels.index(label) for label in labels]

    # And next making the lists to numpy array
    images = np.array(images)
    integer_labels = np.array(integer_labels)

    # And in our task we need to train and valid then test it, so the best solution for that ->
    # -> is the make mix of all images in one group, then take 90% for train and 10% for test
    mix = np.random.permutation(len(images))
    images = images[mix]
    int_labels = integer_labels[mix]

    # Now we r make for testing 10% of images
    test_size = int(len(images) * 0.1)  # 10% for testing
    X_dataset_test = images[:test_size]
    Y_dataset_test = int_labels[:test_size]

    # and now here the last 90% for training and validation
    X_dataset_traing = images[test_size:]
    Y_dataset_traing = int_labels[test_size:]

    return X_dataset_traing, Y_dataset_traing, X_dataset_test, Y_dataset_test, new_labels