import argparse
from collections import namedtuple
from pathlib import Path

import json
from tqdm import tqdm
from PIL import Image

IMAGE = namedtuple("image", "image_id path width height labels")
ANNOTATION = namedtuple("annotation", "image_id XMin,XMax,YMin,YMax")


def normalize(
    thi_dataset,
    json_file_path,
    to_mscoco,
    to_openimages,
):

    if to_mscoco:
        normalize_thi(
            thi_dataset,
            json_file_path,
        )
    elif to_openimages:
        normalize_coco(
            thi_dataset,
            json_file_path,
        )
    else:
        print("either set --to-coc or --to-openimages")


def normalize_thi(
    thi_dataset,
    json_file_path,
):
    annotation_data = {
        "annotations": None,
        "categories": None,
        "images": None,
        "info": None,
        "licenses": "THI",
    }

    categories = []
    categories_dict = {}

    classes = [[0, "Vehicle registration plate"]]

    for idx, item in tqdm(classes, ascii=True, total=len(classes)):

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
        "description": "THI",
        "url": "https://www.thi.de/forschung/carissma/c-isafe/thi-license-plate-dataset",
        "version": 1,
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

    samples = list(thi_dataset.iterdir())
    dataset_images = []
    dataset_annotations = []

    img_idx = 0
    ann_idx = 0

    for sample in tqdm(samples, ascii=True, unit="types"):
        days = list(sample.iterdir())
        for day in tqdm(days, ascii=True, unit="days", leave=False, total=len(days)):
            images = list(day.iterdir())
            images = [image for image in images if image.suffix == ".jpg"]
            annotation_file = day.joinpath(day.name + ".txt")
            with annotation_file.open("r") as pfile:
                annotations = pfile.readlines()
            ann_list = [ann.strip() for ann in annotations]

            for ann in tqdm(ann_list, ascii=True, unit="annotations", leave=False):
                ann = ann.split(";")[:-1]
                img, anno = ann[0], ann[1:]

                img_path = day.joinpath(img).as_posix()
                pil_img = Image.open(img_path)

                img = {
                    "license": 4,
                    "file_name": f"{img_path.split(thi_dataset.as_posix())[1][1:]}",
                    "coco_url": "TBD",
                    "height": pil_img.height,
                    "width": pil_img.width,
                    "date_captured": "UNK",
                    "flickr_url": "UNK",
                    "id": img_idx,
                }
                dataset_images.append(img)

                for box in anno:
                    box = list(map(int, map(float, box.split(","))))
                    x_min, y_min, width, height = box
                    box_ann = {
                        "area": width * height,
                        "bbox": [x_min, y_min, width, height],
                        "category_id": 1,
                        "id": ann_idx,
                        "image_id": img_idx,
                        "iscrowd": 0,
                        "segmentation": "UNK",
                    }
                    dataset_annotations.append(box_ann)
                    ann_idx += 1
                img_idx += 1

    annotation_data["images"] = dataset_images
    print(f"Done adding {len(dataset_images)} images")

    annotation_data["annotations"] = dataset_annotations
    print(f"Done adding {len(dataset_annotations)} annotations")

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
        "Normalize THI <—> MSCOCO for training EfficientDet"
    )
    parser.add_argument(
        "-t",
        "--thi",
        dest="thi_dataset",
        help="THI Dataset path",
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
        args.thi_dataset,
        args.json_path,
        args.to_mscoco,
        args.to_openimages,
    )
