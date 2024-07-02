import os
from PIL import Image

def resize_image(image_path, size=(48, 48)):
    try:
        with Image.open(image_path) as img:
            img_resized = img.resize(size, Image.LANCZOS)
            img_resized.save(image_path)
            print(f'Resized and saved image: {image_path}')
    except Exception as e:
        print(f'Error processing image {image_path}: {e}')

def process_directory(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(dirpath, filename)
                resize_image(image_path)

if __name__ == "__main__":
    root_directory = 'data'
    process_directory(root_directory)

