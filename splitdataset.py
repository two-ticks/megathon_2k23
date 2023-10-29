import os
import shutil
import random
dataset_dir = 'k:/megathon/paddy-disease-classification/train_images'  # Change this to your dataset directory
train_dir = 'k:/megathon/paddy-disease-classification/train'      # Destination for the training data
val_dir = 'k:/megathon/paddy-disease-classification/validation'    # Destination for the validation data

split_ratio = 0.9

# Create the destination directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Iterate through the class directories in the dataset
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)

    # Get a list of all files in the class directory
    files = os.listdir(class_dir)

    # Shuffle the list of files for randomness
    random.shuffle(files)

    # Calculate the split point
    split_point = int(split_ratio * len(files))

    # Split the files into training and validation
    train_files = files[:split_point]
    val_files = files[split_point:]

    # Copy files to the corresponding directories
    for file in train_files:
        src = os.path.join(class_dir, file)
        dest = os.path.join(train_dir, class_name, file)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(src, dest)

    for file in val_files:
        src = os.path.join(class_dir, file)
        dest = os.path.join(val_dir, class_name, file)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(src, dest)
