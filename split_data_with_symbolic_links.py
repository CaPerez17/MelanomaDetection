import os
import random
import shutil

# Define paths
data_dir = '/path/to/data'
train_dir = '/path/to/train'
val_dir = '/path/to/val'
test_dir = '/path/to/test'

# Create directories for train, validation, and test sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define train/validation/test split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Get list of all image files
image_files = os.listdir(os.path.join(data_dir, 'images'))

# Shuffle the list of image files
random.shuffle(image_files)

# Calculate number of images for each split
num_train = int(len(image_files) * train_ratio)
num_val = int(len(image_files) * val_ratio)
num_test = len(image_files) - num_train - num_val

# Loop through image files and save them to the appropriate directories using symbolic links
for i, image_file in enumerate(image_files):
    # Create symbolic link to image file
    src = os.path.join(data_dir, 'images', image_file)
    if i < num_train:
        dst = os.path.join(train_dir, image_file)
    elif i < num_train + num_val:
        dst = os.path.join(val_dir, image_file)
    else:
        dst = os.path.join(test_dir, image_file)
    os.symlink(src, dst)
    
    # Save corresponding label to text file
    label = 1 if image_file.startswith('melanoma') else 0
    if i < num_train:
        label_file = os.path.join(train_dir, image_file[:-4] + '.txt')
    elif i < num_train + num_val:
        label_file = os.path.join(val_dir, image_file[:-4] + '.txt')
    else:
        label_file = os.path.join(test_dir, image_file[:-4] + '.txt')
    with open(label_file, 'w') as f:
        f.write(str(label))
