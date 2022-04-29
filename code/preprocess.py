# install pillow library
# python3 -m pip install --upgrade pip
# python3 -m pip install --upgrade Pillow

from PIL import Image
import glob, os

def preprocess(path):
    size = 256, 256
    count = 1

    for infile in glob.glob(path + "/*.jpg"):
        #file, ext = os.path.splitext(infile)
        with Image.open(infile) as im:
            im.thumbnail(size)
            im.save('img' + str(count) + '.jpg', "JPEG")
            count += 1

def main():
    preprocess('skin_image_data')

if __name__ == '__main__':
    main()