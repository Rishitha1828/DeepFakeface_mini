import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to the datasets
data1_real_dir = 'C:/minipro/proj/140k deep/real_and_fake_face/training_real'
data1_fake_dir = 'C:/minipro/proj/140k deep/real_and_fake_face/training_fake'
data2_real_dir = 'C:/minipro/proj/real vs fake deep/real_vs_fake/real-vs-fake/train/real'
data2_fake_dir = 'C:/minipro/proj/real vs fake deep/real_vs_fake/real-vs-fake/train/fake'

# Path for the merged dataset
merged_dataset_dir = 'C:/minipro/proj/combined-real-and-fake-faces/combined-real-vs-fake'
train_dir = os.path.join(merged_dataset_dir, 'train')
valid_dir = os.path.join(merged_dataset_dir, 'valid')
test_dir = os.path.join(merged_dataset_dir, 'test')

# Create directories for the merged dataset
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for label in ['real', 'fake']:
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

# Function to copy images to their respective directories
def copy_images(src_dirs, filenames, dest_dir):
    for src_dir, filename in filenames:
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dest_dir, filename)
        shutil.copy(src, dst)

# Load file names
data1_real_files = os.listdir(data1_real_dir)
data1_fake_files = os.listdir(data1_fake_dir)
data2_real_files = os.listdir(data2_real_dir)
data2_fake_files = os.listdir(data2_fake_dir)

# Combine file lists
real_files = [(data1_real_dir, f) for f in data1_real_files] + [(data2_real_dir, f) for f in data2_real_files]
fake_files = [(data1_fake_dir, f) for f in data1_fake_files] + [(data2_fake_dir, f) for f in data2_fake_files]

# Create labels
real_labels = ['real'] * len(real_files)
fake_labels = ['fake'] * len(fake_files)

# Combine files and labels
all_files = real_files + fake_files
all_labels = real_labels + fake_labels

# Split the data into train, validation, and test sets
train_files, temp_files, train_labels, temp_labels = train_test_split(all_files, all_labels, test_size=0.3, stratify=all_labels, random_state=42)
valid_files, test_files, valid_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

# Function to move images from src_dirs to dest_dirs
def move_images(filenames, labels, dest_dirs):
    for (src_dir, filename), label in zip(filenames, labels):
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dest_dirs[label], filename)
        shutil.copy(src, dst)

# Train directories
dest_dirs = {'real': os.path.join(train_dir, 'real'), 'fake': os.path.join(train_dir, 'fake')}
move_images(train_files, train_labels, dest_dirs)

# Validation directories
dest_dirs = {'real': os.path.join(valid_dir, 'real'), 'fake': os.path.join(valid_dir, 'fake')}
move_images(valid_files, valid_labels, dest_dirs)

# Test directories
dest_dirs = {'real': os.path.join(test_dir, 'real'), 'fake': os.path.join(test_dir, 'fake')}
move_images(test_files, test_labels, dest_dirs)

print("Images moved successfully.")
