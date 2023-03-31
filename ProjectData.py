import os
import random
from sklearn.model_selection import train_test_split

# Set path to preprocessed image data
data_dir = "path/to/preprocessed/data"

# Create lists of image filenames and corresponding labels
filenames = os.listdir(data_dir)
labels = [filename.split("_")[0] for filename in filenames]

# Split data into train/val/test sets
train_filenames, test_filenames, train_labels, test_labels = train_test_split(filenames, labels, test_size=0.1, random_state=42)
train_filenames, val_filenames, train_labels, val_labels = train_test_split(train_filenames, train_labels, test_size=0.111, random_state=42)

# Save split data to separate directories
os.makedirs("path/to/train", exist_ok=True)
os.makedirs("path/to/val", exist_ok=True)
os.makedirs("path/to/test", exist_ok=True)

for i in range(len(train_filenames)):
    filename = train_filenames[i]
    label = train_labels[i]
    os.symlink(os.path.join(data_dir, filename), os.path.join("path/to/train", label + "_" + filename))
    
for i in range(len(val_filenames)):
    filename = val_filenames[i]
    label = val_labels[i]
    os.symlink(os.path.join(data_dir, filename), os.path.join("path/to/val", label + "_" + filename))

for i in range(len(test_filenames)):
    filename = test_filenames[i]
    label = test_labels[i]
    os.symlink(os.path.join(data_dir, filename), os.path.join("path/to/test", label + "_" + filename))
