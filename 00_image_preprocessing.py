import os
from PIL import Image


def image_resize(input_dir, output_dir, desired_size=(256, 256), extension='.bmp'):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(extension):
            bmp_path = os.path.join(input_dir, filename)
            try:
                with Image.open(bmp_path) as img:
                    # Provjera načina slike (mode)
                    if img.mode == 'L':
                        # Slika je grayscale (mask)
                        resample = Image.NEAREST  # Koristi NEAREST za maske kako se ne bi izgubile oznake
                    else:
                        # Slika je RGB ili drugi color mode
                        resample = Image.BILINEAR  # Koristi BILINEAR za bojne slike

                    # Promjena veličine na željenu veličinu
                    resized_img = img.resize(desired_size, resample)

                    # Definirajte novi naziv s JPG ekstenzijom
                    base_name = os.path.splitext(filename)[0]
                    jpg_filename = f"{base_name}.jpg"
                    jpg_path = os.path.join(output_dir, jpg_filename)

                    # Konvertiranje i spremanje slike
                    if img.mode == 'L':
                        # Spremanje grayscale maske
                        resized_img.save(jpg_path, 'JPEG')
                    else:
                        # Konvertiranje u RGB prije spremanja
                        resized_img.convert('RGB').save(jpg_path, 'JPEG')

                    print(f"Obrađeno: {filename} -> {jpg_filename}")
            except Exception as e:
                print(f"Greška pri obradi {filename}: {e}")

if __name__ == "__main__":

    INPUT_DIR = "raw_images_with_labels"  # Zamijenite s točnom putanjom
    OUTPUT_DIR = "raw_images_256_with_labels"  # Zamijenite s točnom putanjom

    image_resize(INPUT_DIR, OUTPUT_DIR, desired_size=(256, 256), extension='.bmp')