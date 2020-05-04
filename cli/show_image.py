import argparse
from IPython.display import display
from PIL import Image
import json
import urllib.request
import requests

TEXTVQA_VAL_JSON = 'https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json'

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image_id",
                    help="refer to image_id from val dataset of open_images, type all to process all images",
                    nargs='+')
args = parser.parse_args()

# fetch image urls
with urllib.request.urlopen(TEXTVQA_VAL_JSON, timeout=1) as url:
    validation_images_data = json.loads(url.read().decode())

images_url = dict()
seen_images = []
for image in validation_images_data['data']:
    if image['image_id'] in args.image_id and image['image_id'] not in seen_images:
        seen_images.append(image['image_id'])
        print("Image_id: ", image['image_id'])
        path = requests.get(image['flickr_300k_url'], stream=True).raw
        #image = Image.open(path)
        path = urllib.request.urlopen(image['flickr_300k_url'])
        with Image(file=path) as image:
            wand.display.display(image)
        # display(image)
        # image.show()


