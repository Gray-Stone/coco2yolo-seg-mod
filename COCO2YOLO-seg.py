#! /usr/bin/python3
"""
Author: Zubin Bhuyan
Date: June 21, 2023

MIT License

"""

import json
import os
import pathlib
import argparse

import dataclasses
from typing import Any
import yaml
import enum
import random

LABEL_FOLDER_NAME = "labels"
IMAGES_FOLDER_NAME = "images"


class ImageType(enum.Enum):
    VAL = "val"
    TRAIN = "train"


@dataclasses.dataclass()
class ImageInfo():

    id : Any
    source_path : pathlib.Path
    height : int
    width : int
    type : ImageType

    def __init__(self, img_dict: dict[Any, Any], data_dir: pathlib.Path) -> None:

        self.id = img_dict['id']
        self.source_path: pathlib.Path = data_dir.absolute() / img_dict['file_name']
        self.height = img_dict['height']
        self.width = img_dict['width']
        self.type = ImageType.TRAIN

    def SetType(self, new_type):
        self.type = new_type

    def GetStem(self):
        return self.source_path.stem

    def GetTypedLabelName(self):
        return pathlib.Path(self.type.value) / (self.source_path.stem + ".txt")

    def GetTypedImageName(self):
        return pathlib.Path(self.type.value) / self.source_path.name

    # def GetLabelPath(self, dest: pathlib.Path):
    #     file_base = dest / LABEL_FOLDER_NAME / str(self.type) / self.file_name.stem
    #     return file_base.with_suffix("txt")

    # def GetImagePath(self, dest: pathlib.Path):
    #     return dest / IMAGES_FOLDER_NAME / str(self.type) / self.file_name.name


class YoloDataset():

    def __init__(self, out_dir: pathlib.Path) -> None:
        self.root_dir: pathlib.Path = out_dir.absolute() # Keep the symlink.
        print(f"output directory root: {self.root_dir}")
        if self.root_dir.exists():
            if (not self.root_dir.is_dir()):
                raise ValueError(f"{self.root_dir} exists and is not a directory")

        for type in ImageType:
            for obj in [LABEL_FOLDER_NAME, IMAGES_FOLDER_NAME]:
                dir = self.root_dir / obj / type.value
                dir.mkdir(parents=True, exist_ok=True)

    def GetImageDest(self, image: ImageInfo):
        return self.root_dir / IMAGES_FOLDER_NAME / image.GetTypedImageName()

    def GetLabelDest(self, image: ImageInfo):
        return self.root_dir / LABEL_FOLDER_NAME / image.GetTypedLabelName()

    def TryLinkImage(self , image:ImageInfo):
        dest = self.GetImageDest(image)
        if not dest.exists():
            dest.symlink_to(image.source_path)

    def GenYaml(self,class_dict):
        yaml_dict = {
            "train" : f"./{IMAGES_FOLDER_NAME}/{ImageType.TRAIN.value}",
            "val" : f"./{IMAGES_FOLDER_NAME}/{ImageType.VAL.value}",
            "path" : str(self.root_dir),
            "names" : class_dict
        }
        with open( self.root_dir / "dataset.yaml" , "w" ) as file:
            yaml.dump(yaml_dict , file )

def convert_coco_to_yolo_segmentation(json_file:pathlib.Path, source_img_dir: pathlib.Path,
                                      dest_folder: pathlib.Path):

    # Load the JSON file
    with open(json_file, 'r') as file:
        coco_data = json.load(file)

    # Extract annotations from the COCO JSON data
    annotations = coco_data['annotations']

    # Create a "labels" folder to store YOLO segmentation annotations
    # output_folder = os.path.join(os.path.dirname(json_file), dest_folder)
    # os.makedirs(output_folder, exist_ok=True)
    yolo_dataset = YoloDataset(dest_folder)
    image_info_dict : dict[Any,ImageInfo] = {}
    for img in coco_data['images']:
        info = ImageInfo(img, source_img_dir)
        image_info_dict[info.id] = info

    # Build the class id and name, used later in final export
    full_category_dict = {}
    for category in coco_data["categories"]:
        full_category_dict[category["id"]] = category["name"]
    yolo_dataset.GenYaml(full_category_dict)

    # Build a thing of images, and decide their destination
    total_img_length = len(coco_data['images'])
    val_size = total_img_length * 0.1  # 10 percent validation
    if total_img_length > 5000:
        val_size = total_img_length * 0.01
    elif total_img_length > 20000:
        val_size = total_img_length * 0.001
    val_size = int(val_size)
    # Set images.
    for val_img in random.sample(list(image_info_dict), val_size):
        image_info_dict[val_img].SetType(ImageType.VAL)

    for info in image_info_dict.values():
        # Make a symlink for the image as well while we loop
        yolo_dataset.TryLinkImage(info)


    for annotation in annotations:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        segmentation = annotation['segmentation']
        bbox = annotation['bbox']

        image_info = image_info_dict[image_id]

        # # This is a not used bbox feature.
        # # Calculate the normalized center coordinates and width/height
        # x_center = (bbox[0] + bbox[2] / 2) / image_info.width
        # y_center = (bbox[1] + bbox[3] / 2) / image_info.height
        # bbox_width = bbox[2] / image_info.width
        # bbox_height = bbox[3] / image_info.height

        # Find the image filename from the COCO data
        # TODO optimize later.
        # for image in coco_data['images']:
        #     if image['id'] == image_id:
        #         image_filename = os.path.basename(image['file_name'])
        #         image_filename = os.path.splitext(image_filename)[
        #             0]  # Removing the extension. (In our case, it is the .jpg or .png part.)
        #         image_width = image['width']
        #         image_height = image['height']
        #         break

        # Convert COCO segmentation to YOLO segmentation format
        yolo_segmentation = [
            f"{(x) / image_info.width:.5f} {(y) / image_info.height:.5f}"
            for x, y in zip(segmentation[0][::2], segmentation[0][1::2])
        ]
        #yolo_segmentation.append(f"{(segmentation[0][0]) / image_width:.5f} {(segmentation[0][1]) / image_height:.5f}")
        yolo_segmentation = ' '.join(yolo_segmentation)

        # Generate the YOLO segmentation annotation line
        yolo_annotation = f"{category_id} {yolo_segmentation}"

        # Also record this category line.
        # Save the YOLO segmentation annotation in a file
        label_filepath = yolo_dataset.GetLabelDest(image_info)
        with open(label_filepath, 'a+') as file:
            file.write(yolo_annotation + '\n')

    print(f"Conversion completed. YOLO segmentation annotations saved in {dest_folder} folder.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=pathlib.Path, help="path to coco json file")
    parser.add_argument("image_dir", default='./data', type=pathlib.Path, help="output dir")
    parser.add_argument("output_dir", default='./Yolo-conv', type=pathlib.Path, help="output dir")

    args = parser.parse_args()
    json_file: pathlib.Path = args.json_file
    out_dir = args.output_dir
    image_source :pathlib.Path = args.image_dir

    if not image_source.exists():
        raise ValueError(f"Image dir {image_source} doesn't exists")

    convert_coco_to_yolo_segmentation(json_file.resolve(), image_source, out_dir)
