from PIL import Image
import os

folder_path = ('/Users/yorkgong/Downloads/Laser_Photo')
output_folder = ("/Users/yorkgong/Downloads/Laser_Photo_PNG")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(folder_path):
    if filename.lower().endswith('.jpg'):
        jpg_file_path = os.path.join(folder_path, filename)

        with Image.open(jpg_file_path) as img:
            png_file_name = os.path.splitext(filename)[0] + '.png'
            png_file_path = os.path.join(output_folder, png_file_name)

            img.save(png_file_path, 'PNG')

            print(f"{filename} has been turned to {png_file_name}")