import argparse
from collections import namedtuple
from pathlib import Path

import json
from tqdm import tqdm
import pandas as pd

IMAGE = namedtuple("image", "image_id path width height labels")
ANNOTATION = namedtuple("annotation", "image_id XMin,XMax,YMin,YMax")


def normalize(
    info_csv,
    bbox_csv,
    classes_csv,
    json_file_path,
    to_mscoco,
    to_openimages,
):

    if to_mscoco:
        normalize_openimages(
            info_csv,
            bbox_csv,
            classes_csv,
            json_file_path,
        )
    elif to_openimages:
        normalize_coco(
            info_csv,
            bbox_csv,
            classes_csv,
            json_file_path,
        )
    else:
        print("either set --to-coc or --to-openimages")


def normalize_openimages(
    info_csv,
    bbox_csv,
    classes_csv,
    json_file_path,
):
    annotation_data = {
        "annotations": None,
        "categories": None,
        "images": None,
        "info": None,
        "licenses": "Google Inc",
    }

    classes = pd.read_csv(classes_csv.as_posix(), header=None)
    print(f"Done reading {classes_csv}")
    infos = pd.read_csv(info_csv.as_posix())
    print(f"Done reading {info_csv}")
    bboxes = pd.read_csv(bbox_csv.as_posix())
    print(f"Done reading {bbox_csv}")

    categories = []
    categories_dict = {}

    for idx, item in tqdm(classes.iterrows(), ascii=True, total=len(classes)):

        index, name = item[0], item[1]

        d = {
            "supercategory": "PII",
            "id": idx + 1,
            "name": name,
            "machine_name": index,
        }
        categories_dict[index] = {"id": idx + 1, "name": name}
        categories.append(d)

    print(f"Done adding {len(categories)} categories")

    annotation_data["categories"] = categories
    info = {
        "description": "OpenImagesV6",
        "url": "https://storage.googleapis.com/openimages/web/index.html",
        "version": 6,
        "year": 2020,
    }
    annotation_data["info"] = info
    licenses = [
        {
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "id": 0,
            "name": "CC-BY-4.0",
        },
        {
            "url": "https://creativecommons.org/licenses/by/2.0/",
            "id": 1,
            "name": "CC-BY-2.0",
        },
    ]
    annotation_data["licenses"] = licenses

    print("Done adding licenses")

    image_ids = infos["id"]
    image_widths = infos["width"]
    image_heights = infos["height"]

    images = []
    images_dict = {}

    for idx, (id, width, height) in tqdm(
        enumerate(zip(image_ids, image_widths, image_heights)),
        unit="images",
        ascii=True,
        total=len(image_ids),
    ):

        img = {
            "license": 4,
            "file_name": f"{id}.jpg",
            "coco_url": "TBD",
            "height": int(height),
            "width": int(width),
            "date_captured": "UNK",
            "flickr_url": "UNK",
            "id": idx,
        }
        images.append(img)
        images_dict[id] = {"width": int(width), "height": int(height), "id": idx}

    annotation_data["images"] = images

    print(f"Done adding {len(images)} images")

    annotations = []

    for idx, box in tqdm(
        bboxes.iterrows(), unit="annotatations", ascii=True, total=len(bboxes)
    ):

        image_id, label = box["ImageID"], box["LabelName"]
        x_min, x_max = box["XMin"], box["XMax"]
        y_min, y_max = box["YMin"], box["YMax"]
        img_width = images_dict[image_id]["width"]
        img_height = images_dict[image_id]["height"]

        x_min = int(x_min * img_width)
        x_max = int(x_max * img_width)
        y_min = int(y_min * img_height)
        y_max = int(y_max * img_height)

        obj_height = y_max - y_min + 1
        obj_width = x_max - x_min + 1
        category_id = categories_dict[label]["id"]

        ann = {
            "area": obj_height * obj_width,
            "bbox": [x_min, y_min, obj_width, obj_height],
            "category_id": category_id,
            "id": idx,
            "image_id": images_dict[image_id]["id"],
            "iscrowd": 0,
            "segmentation": "UNK",
        }

        annotations.append(ann)

    annotation_data["annotations"] = annotations
    print(f"Done adding {len(annotations)} annotations")

    with json_file_path.open("w") as pfile:

        json.dump(annotation_data, pfile)

    print(f"Done writing {json_file_path}")


def normalize_coco(
    info_csv,
    bbox_csv,
    labels_csv,
    classes_csv,
    json_file_path,
    class_txt,
):

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Normalize OpenImagesV6 <—> MSCOCO for training EfficientDet"
    )
    parser.add_argument(
        "-i",
        "--info",
        dest="info_csv",
        help="Dataset Info csv",
        type=Path,
    )
    parser.add_argument(
        "-b",
        "--bbox",
        dest="bbox_csv",
        help="Dataset BBOX csv",
        type=Path,
    )
    parser.add_argument(
        "-x",
        "--class_description",
        dest="classes_csv",
        help="Datset Classes csv",
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--destination",
        type=Path,
        dest="json_path",
        help="Json Path",
    )
    parser.add_argument(
        "--to-mscoco",
        action="store_true",
        default=True,
        dest="to_mscoco",
        help="Convert to MSCOCO style data format",
    )
    parser.add_argument(
        "--to-openimages",
        action="store_true",
        default=False,
        dest="to_openimages",
        help="Convert to OpenImagesV6 style data format",
    )
    args = parser.parse_args()

    normalize(
        args.info_csv,
        args.bbox_csv,
        args.classes_csv,
        args.json_path,
        args.to_mscoco,
        args.to_openimages,
    )
