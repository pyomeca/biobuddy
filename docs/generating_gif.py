from PIL import Image
import os

png_path = "BioBuddy_images"
png_files = sorted([file for file in os.listdir(png_path) if file.endswith('.png')])

# Open images
frames = [Image.open(png_path + "/" + png) for png in png_files]

# Save as GIF
frames[0].save(
    "biobuddy.gif",
    format='GIF',
    save_all=True,
    append_images=frames[1:],
    duration=25,  # duration per frame in milliseconds
    loop=0         # loop forever
)
