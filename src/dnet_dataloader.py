import glob
from PIL import Image
import random
import re
import json

import torch
from torch.utils.data import Dataset


# #### The code from last post
POINTS = re.compile('POLYGON\s*\(\((.+)\)\)', re.I)

def polygon(building):
    poly = building['wkt']
    poly = POINTS.match(poly).group(1)
    poly = [coord.split(' ') for coord in poly.split(', ')]
    poly = [(float(x), float(y)) for x, y in poly]

    return poly

# Code from this post
def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value

def bbox(poly, padding = 20):
    xmin, ymin = poly[0]  # top-left
    xmax, ymax = poly[0]  # bottom-right

    for x, y in poly[1:]:
        if x < xmin:
            xmin = x
        elif x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        elif y > ymax:
            ymax = y

    xmin -= padding
    ymin -= padding

    xmax += padding
    ymax += padding

    return [(xmin, ymin), (xmax, ymax)]

def norm_bbox(bbox, image_width, image_height = None):
    if not image_height:
        image_height = image_width

    xmin, ymin = bbox[0] # top-left
    xmax, ymax = bbox[1] # bottom-right

    # Clip x-values
    xmin = clip(xmin, 0, image_width)
    xmax = clip(xmax, 0, image_width)

    # Clip y-values
    ymin = clip(ymin, 0, image_height)
    ymax = clip(ymax, 0, image_height)

    return [(xmin, ymin), (xmax, ymax)]


class DamageNetDataset(Dataset):
    """xView2 dataset.
    Parameters
    ----------
    images_dir: str
        Directory with all the images.
    labels_dir: str
        Directory with all the labels as JSON files.
    transform: callable, optional
        Optional transform to be applied on a sample. Transform receives
        the current image **and** target! Default is `None`.
    """

    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_list = sorted(glob.glob(images_dir + '/*_post_disaster.png'))
        self.labels_list = sorted(glob.glob(labels_dir + '/*_post_disaster.json'))

        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Check to see if buildings is None
        with open(self.labels_list[idx]) as f:
            label = json.load(f)
            buildings = label['features']['xy']

            # Choose a random builing and get its damage_value
            if buildings:
                chosen_building = random.choice(buildings)
                damage_value = chosen_building['properties']['subtype']

        while not buildings or damage_value == 'un-classified':
            idx = random.randint(0, self.__len__())

            with open(self.labels_list[idx]) as f:
                label = json.load(f)
                buildings = label['features']['xy']

                # Choose a random builing and get its damage_value
                if buildings:
                    chosen_building = random.choice(buildings)
                    damage_value = chosen_building['properties']['subtype']

        # Do get_data with working values
        image, label = self.get_data(idx, chosen_building)

        return image, label

    def get_data(self, idx, chosen_building):
        image = Image.open(self.images_list[idx])

        coords = polygon(chosen_building)
        box = bbox(coords)
        box = norm_bbox(box, image.width, image.height)
        cropped_bbox_image = image.crop((box[0][0], box[0][1], box[1][0], box[1][1]))
        # plt.imshow(cropped_bbox_image)
        # plt.show()

        damage_value = chosen_building['properties']['subtype']
        # print(damage_value)

        # Convert str in damage_value to a pytorch tensor
        if damage_value == 'no-damage':
            label = torch.Tensor([0, 0, 0])
        elif damage_value == 'minor-damage':
            label = torch.Tensor([1, 0, 0])
        elif damage_value == 'major-damage':
            label = torch.Tensor([1, 1, 0])
        elif damage_value == 'destroyed':
            label = torch.Tensor([1, 1, 1])

        image = cropped_bbox_image

        if self.transform:
            image = self.transform(image)

        return image, label
