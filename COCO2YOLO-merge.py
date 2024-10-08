#! /usr/bin/python3
"""
Author: Leo Chen 
Date: 2024-06-19T23:45:23Z-05

MIT License

This is a script to merge additional training data in coco format into existing YOLO dataset.

Requirement: 
    * Existing YOLO dataset.yaml file with all needed classes listed.
      json file for coco labels. 
    * The images for coco dataset are expected to already exists in the original yolo dataset. 
      (could be placed on side, unsorted)
    * Image's filename stem is used as it's unique ID. It is used to associate images from coco's record.


This script is made with the intension for merging additional labels made on existing per-labeled
dataset. Thus it have expectation on image files all previously exists in yolo's dataset folder.

If a new label in coco's json associated to an image without existing label files, a new file 
will be created and placed right next to the existing image, dis-regarding the `image` vs `label` 
parent folder since YOLO can figure things out either way.
"""

import json
import pathlib
import argparse

import dataclasses
from typing import Any
import yaml
import enum

LABEL_FOLDER_NAME = "labels"
IMAGES_FOLDER_NAME = "images"


class ImageType(enum.Enum):
    VAL = "val"
    TRAIN = "train"


@dataclasses.dataclass()
class ImageInfo():

    id : Any
    # source_path : pathlib.Path
    height : int
    width : int
    type : ImageType

    def __init__(self, img_dict: dict[Any, Any]) -> None:

        self.id = img_dict['id']
        self.stem_id = pathlib.Path(img_dict['file_name']).stem

        self.height = img_dict['height']
        self.width = img_dict['width']
        self.type = ImageType.TRAIN

    def SetType(self, new_type):
        self.type = new_type

class YoloDataset():

    def __init__(self, dest_yaml: pathlib.Path) -> None:
        self.root_dir: pathlib.Path = dest_yaml.absolute().parent # Keep the symlink.
        print(f"output directory root: {self.root_dir}")
        if self.root_dir.exists():
            if (not self.root_dir.is_dir()):
                raise ValueError(f"{self.root_dir} exists and is not a directory")

        dataset_content = {}
        with open(dest_yaml , "r") as file:
            dataset_content = yaml.safe_load(file)

        print(f"Dataset content: \n{dataset_content}")
        self.class_lists = dataset_content["names"]

        # This is to find ID with class name, we don't use coco's class ID, 
        # only use class name to link classes.
        self.class_name_id_dict = {}
        for id , cls_name in self.class_lists.items():
            self.class_name_id_dict[cls_name] = id
        print(f"Flipped look up: {self.class_name_id_dict}")


    def AddLabel(self,image_id:str, new_label_line:str) -> bool:
        """Try to add new lines to existing label file, or create new one if only image file exists

        Args:
            image_id (str): stem of image file name, which is used as ID
            new_label_line (str): Content to be added as a new line to YOLO

        Returns:
            bool: Weather it is successfully added, or even image file is not found
        """

        # Search for the image's label file.
        maybe_file = self.find_label_file(image_id)
        if maybe_file:
            # TODO check file for dupe!
            with open(maybe_file,"a") as f :
                f.write(new_label_line + "\n")
                return True

        maybe_image = self.find_image(image_id)
        if maybe_image:
            label_file = maybe_image.with_suffix(".txt")
            with open(label_file,"w") as f :
                f.write(new_label_line + "\n")
                return True

        return False

    def find_label_file(self,id):
        maybe_files =list(self.root_dir.glob(f"**/{id}.txt"))
        if maybe_files:
            return maybe_files[0]
        else:
            return None

    def find_image(self,id):
        maybe_files =list(self.root_dir.glob(f"**/{id}.*"))
        if not maybe_files:
            return None
        file :pathlib.Path = None
        for file in maybe_files:
            if file.suffix.lower() == ".txt":
                continue
            return file
        print(f"!! image id {id} exists without potential image file.")
        return None


def convert_coco_to_yolo_segmentation(json_file:pathlib.Path,
                                      dest_yaml: pathlib.Path,action_bbox = False):

    # Load the JSON file
    with open(json_file, 'r') as file:
        coco_data = json.load(file)

    # Extract annotations from the COCO JSON data
    annotations = coco_data['annotations']

    # Load in the existing yolo information.
    yolo_dataset = YoloDataset(dest_yaml)

    # Load all image's information from coco's json.
    # We only care about size info, since we are not merging new images.
    image_info_dict : dict[Any,ImageInfo] = {}
    for img in coco_data['images']:
        info = ImageInfo(img)
        image_info_dict[info.id] = info

    # Build the class id and name, used later in final export
    input_category_dict = {}
    for category in coco_data["categories"]:
        input_category_dict[category["id"]] = category["name"]

    # Looping through all segmentation. 
    merged_count = 0
    for annotation in annotations:
        image_id = annotation['image_id']
        source_class_id = annotation['category_id']
        segmentation = annotation['segmentation']
        bbox = annotation['bbox']

        source_class_name = input_category_dict[source_class_id]
        out_class_id = yolo_dataset.class_name_id_dict[source_class_name]


        image_info = image_info_dict[image_id]

        if action_bbox:
            # This is a not used bbox feature.
            # Calculate the normalized center coordinates and width/height
            x_center = (bbox[0] + bbox[2] / 2) / image_info.width
            y_center = (bbox[1] + bbox[3] / 2) / image_info.height
            bbox_width = bbox[2] / image_info.width
            bbox_height = bbox[3] / image_info.height
            # output layout class x_center y_center width height
            yolo_annotation = f"{out_class_id} {x_center} {y_center} {bbox_width} {bbox_height}"

        else:
            if not segmentation:
                # skip empty segmentation, some image might only have bbox but no segmentation.
                continue

            # Convert COCO segmentation to YOLO segmentation format
            yolo_segmentation = [
                f"{(x) / image_info.width:.5f} {(y) / image_info.height:.5f}"
                for x, y in zip(segmentation[0][::2], segmentation[0][1::2])
            ]
            yolo_segmentation = ' '.join(yolo_segmentation)

            # Generate the YOLO segmentation annotation line
            yolo_annotation = f"{out_class_id} {yolo_segmentation}"

        # For debug 
        # print(f"Image {image_info.stem_id} , adding {yolo_annotation}")

        # Save the YOLO segmentation annotation in a file
        if not yolo_dataset.AddLabel(image_info.stem_id ,yolo_annotation):
            # TODO, if need to allow adding image to original dataset, could add lines here. 
            print(f"Cannot add values for {image_id}, no label or image file found")
        else:
            merged_count +=1

    print(f"Merge completed. {merged_count} label added.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=pathlib.Path, help="path to coco json file")
    # parser.add_argument("image_dir", default='./data', type=pathlib.Path, help="output dir")
    parser.add_argument("dest_yaml", default='./Yolo-conv', type=pathlib.Path, help="output dataset yaml")
    parser.add_argument("--bbox" , default=False , action="store_true" , help="Default doing segmentation, if set, then do bounding box.")

    args = parser.parse_args()
    json_file: pathlib.Path = args.json_file
    dest_yaml = args.dest_yaml


    convert_coco_to_yolo_segmentation(json_file.resolve(), dest_yaml , action_bbox = args.bbox)
