from PIL import Image
import numpy as np
import os

# Define the paths to the input and output directories
input_dir = '/home1/abhinav/aml/aml_project/datasets'
output_dir = './FashionMNIST/train/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the data from the IDX file
with open(os.path.join(input_dir, 'train-images-idx3-ubyte'), 'rb') as f:
    magic_number = int.from_bytes(f.read(4), 'big')
    # second 4 bytes is the number of images
    image_count = int.from_bytes(f.read(4), 'big')
    # third 4 bytes is the row count
    row_count = int.from_bytes(f.read(4), 'big')
    # fourth 4 bytes is the column count
    column_count = int.from_bytes(f.read(4), 'big')
    # rest is the image pixel data, each pixel is stored as an unsigned byte
    # pixel values are 0 to 255
    image_data = f.read()
    train_images = np.frombuffer(image_data, dtype=np.uint8)\
        .reshape((image_count, row_count, column_count))

# Loop over all images and save them as PNG files
for i in range(image_count):
    output_path = os.path.join(output_dir, f'image_{i}.png')
    with Image.fromarray(train_images[i]) as im:
        im.save(output_path)
