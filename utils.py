import cv2
import os
import random
import numpy as np

def read_image(path):
    """
    Read the image file given the path
    """
    img = cv2.imread(path) # cv2.imread originally returns image with BGR channels (Blue-Green-Red)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converts (Blue-Green-Red) --> (Red-Green-Blue)
    return img


def split_data(dir, split = 0.9):
    train_set, test_set = {}, {}
    folders = os.listdir(dir)

    num_train_samples = int(len(folders) * split)
    random.shuffle(folders)

    for train_folder in folders[:num_train_samples]:
        num_training_files = len(os.listdir(os.path.join(dir, train_folder)))
        train_set[train_folder] = num_training_files

    for test_folder in folders[num_train_samples:-1]:
        num_training_files = len(os.listdir(os.path.join(dir, test_folder)))
        test_set[test_folder] = num_training_files

    return train_set, test_set


def create_triplets(dir, folder_list, max_samples = 10):
    """
        Generates a set of triplets containing (anchor, positive, negative)
        for training Siamese Network with triplet loss
    """
    triplets = []
    folder_ids = list(folder_list.keys())

    for folder_id in folder_ids:
        folder_path = os.path.join(dir, folder_id)
        files = os.listdir(folder_path)[: max_samples] # Select a set of files given a max number of samples
        num_files = len(files) # List number of files in each dir

        # Loop thru each item in each person face directory
        for i in range(num_files-1):
            for j in range(i+1, num_files):
                anchor = (folder_path, f"{i}.jpg")
                positive = (folder_path, f"{j}.jpg")

                neg_folder = folder_path

                # Randomly select a person face directory that is not the same to anchor
                while neg_folder == folder_path:
                    neg_folder = random.choice(folder_ids)
                
                # Select a file in the negative person face directoory
                neg_file = random.randint(0, folder_list[neg_folder]-1)
                negative = (os.path.join(dir, neg_folder), f"{neg_file}.jpg")

                # Group each as triplets
                triplets.append((anchor, positive, negative))
            
    random.shuffle(triplets)

    return triplets

def preprocess_samples(sample_list):
    """
        Pre-process the data
    """
    return sample_list

def generate_samples(triplet_list, pre_process = True):
    """
    Generate a set of batches for training and testing
    Expected output: (num_samples, 3, img_width, img_height, 3)
    """
    anchor_batch, positive_batch, negative_batch = [], [], []

    # Shuffle the triplet list first
    np.random.shuffle(triplet_list)

    for sample in triplet_list:
        anchor = read_image(os.path.join(sample[0][0], sample[0][1]))
        positive = read_image(os.path.join(sample[1][0], sample[1][1]))
        negative = read_image(os.path.join(sample[2][0], sample[2][1]))
        anchor_batch.append(anchor)
        positive_batch.append(positive)
        negative_batch.append(negative)

    anchor_batch = np.stack(anchor_batch, axis = 0)
    positive_batch = np.stack(positive_batch, axis = 0)
    negative_batch = np.stack(negative_batch, axis = 0)

    if pre_process:
        anchor_batch = preprocess_samples(anchor_batch)
        positive_batch = preprocess_samples(positive_batch)
        negative_batch = preprocess_samples(negative_batch)
    
    batch = [anchor_batch, positive_batch, negative_batch]
    batch = np.stack(batch, axis = 1)

    print(f"batch.shape: {batch.shape}")
    return batch

def generate_batch(triplet_list, batch_size = 256, pre_process = True):
    """
    Generate a set of batches for training and testing
    Expected output: list of numpy array with dimension (batch_size, 3, img_width, img_height, 3)
    """
    num_samples = len(triplet_list)
    num_batches  = int(np.ceil(num_samples / batch_size))

    print(f"num_batches: {num_batches}")
    batches = []

    # Shuffle the triplet list first
    np.random.shuffle(triplet_list)
    anchor_batches, positive_batches, negative_batches = [], [], []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch = triplet_list[start_idx: end_idx]
        anchor_batch, positive_batch, negative_batch = [], [], []

        for j in range(len(batch)):
            batch_j = batch[j]
            anchor = read_image(os.path.join(batch_j[0][0], batch_j[0][1]))
            positive = read_image(os.path.join(batch_j[1][0], batch_j[1][1]))
            negative = read_image(os.path.join(batch_j[2][0], batch_j[2][1]))
            anchor_batch.append(anchor)
            positive_batch.append(positive)
            negative_batch.append(negative)

        anchor_batches.append(np.stack(anchor_batch, axis = 0))
        positive_batches.append(np.stack(positive_batch, axis = 0))
        negative_batches.append(np.stack(negative_batch, axis = 0))

    if pre_process:
        anchor_batches = preprocess_samples(anchor_batches)
        positive_batches = preprocess_samples(positive_batches)
        negative_batches = preprocess_samples(negative_batches)
    
    return anchor_batches, positive_batches, negative_batches