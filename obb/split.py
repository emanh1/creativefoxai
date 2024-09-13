import os
import random
import shutil

train_images_dir = 'datasets/train/images'
train_labels_dir = 'datasets/train/labels'
val_images_dir = 'datasets/val/images'
val_labels_dir = 'datasets/val/labels'

os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

images = os.listdir(train_images_dir)

random.shuffle(images)

split_index = int(len(images) * 0.8)

train_images = images[:split_index]
val_images = images[split_index:]

for image in val_images:
    label = image.replace('.png', '.txt')  

    shutil.move(os.path.join(train_images_dir, image), os.path.join(val_images_dir, image))

    shutil.move(os.path.join(train_labels_dir, label), os.path.join(val_labels_dir, label))

print(f"Moved {len(val_images)} images and labels to validation set.")
