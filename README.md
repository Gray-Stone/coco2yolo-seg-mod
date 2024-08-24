
**This is a fork of repo (https://github.com/z00bean/coco2yolo-seg.git)[https://github.com/z00bean/coco2yolo-seg.git] with my custom edits and addition**

Convert COCO segmentation annotation to YOLO segmentation format with the ".txt per image" format. Does auto val/train segmentation during conversion.

Separate script handles merging additional coco labels into existing yolo project.

This repo only contain 2 simple python scripts, and are expected to be cloned and run by user. The script should be intuitive for modifying to custom needs.

## YOLO Segmentation Data Format


The dataset format used for training YOLO segmentation models is as follows:

- One text file per image: Each image in the dataset has a corresponding text file with the same name as the image file and the ".txt" extension.
- One row per object: Each row in the text file corresponds to one object instance in the image.
- Object information per row: Each row contains the following information about the object instance:
  - Object class index: An integer representing the class of the object (e.g., 0 for person, 1 for car, etc.).
  - Object bounding coordinates: The bounding coordinates around the mask area, normalized to be between 0 and 1.

The format for a single row in the segmentation dataset file is as follows: 
`<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>`

The coordinates are separated by spaces.


## COCO2YOLO-seg.py Usage

Example usage:
```
coco2yolo-seg-mod/COCO2YOLO-seg.py coco_folder/coco_labels.json coco_folder/data output_folder/ --bbox
```

The script `COCO2YOLO-seg.py` took 3 positional arguments
  * coco json path: File path to the coco json files that contain all the labels
  * coco data path: Folder path to the image folder where coco expects things to be at
  * yolo output path: Folder path to place output yolo dataset.yaml and image folder.
  * optional bbox: with this flag set, the script will actually do bbox instead of segmentation

### potential catches:

To save on disk-space, all images are symlinked, not copied. If desired, you can change the source code to copy if necessary. 

Train/Val folder are auto generated and data are auto split into each folder. The split ratio is 
* 10% val for image size below 5000
* 1% val for image size from 5000 to 20k
* 0.1% for image size above 20k 

## COCO2YOLO-merge.py

This script is used for merging additional COCO labeling into existing yolo dataset. 


Usage:
```
coco2yolo-seg/COCO2YOLO-merge.py <source coco json> <destination yolo dataset.yaml> --<bbox=False>
```
  * source coco json: coco's json label file with new labels
  * destination yolo dataset.yaml: destination yolo dataset to merge data into.
    * It is expected that images for this dataset is somewhere in a subfolder from where the dataset.yaml is.
  * optional bbox: with this flag set, the script will actually do bbox instead of segmentation
  
The classes list in yolo dataset are treated as the target class IDs. Only class names are used from coco's data. This way, any id different from coco's dataset is ignored.

For my use case I've picked a small subset from a huge per-labeled dataset and added label for a new classes, then re-added their label back from original dataset. Thus the design actually doesn't allow adding new images from input coco dataset, only allow merging label onto existing YOLO dataset. This way, the sub-set images are not polluted with more images that might contain new class but not labeled.

Another consideration for not adding new images are the difficulty for deciding on val/train. Since YOLO dataset allows a few different way of representing val/train. Thus it is simply not handled.

## License

MIT License. (same with upstream)

