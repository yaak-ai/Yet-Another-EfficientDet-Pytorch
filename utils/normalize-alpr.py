import argparse
from collections import namedtuple
from pathlib import Path

import json
from tqdm import tqdm
from PIL import Image

IMAGE = namedtuple("image", "image_id path width height labels")
ANNOTATION = namedtuple("annotation", "image_id XMin,XMax,YMin,YMax")


def normalize(
    alpr_dir,
    json_file_path,
    to_mscoco,
    to_openimages,
):

    if to_mscoco:
        normalize_alpr(
            alpr_dir,
            json_file_path,
        )
    elif to_openimages:
        normalize_coco(
            alpr_dir,
            json_file_path,
        )
    else:
        print("either set --to-coc or --to-openimages")


def get_annotation(img_name):

    annotation_file = img_name.with_suffix(".txt")

    if not annotation_file.is_file():
        return None
    with annotation_file.open() as pfile:
        info = pfile.readlines()
    bbox = info[0].split()

    bbox = list(map(int, bbox[1:-1]))

    return [bbox[0], bbox[1]], bbox[2], bbox[3]


def normalize_alpr(
    alpr_dir,
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
        "url": "https://github.com/openalpr/benchmarks/tree/master/endtoend",
        "version": 1,
        "year": 2020,
    }
    annotation_data["info"] = info
    licenses = [
        {
            "url": "https://github.com/openalpr/benchmarks/blob/master/LICENSE",
            "id": 0,
            "name": "CC-BY-4.0",
        },
    ]
    annotation_data["licenses"] = licenses

    print("Done adding licenses")

    sample_dir = [a for a in list(alpr_dir.iterdir()) if a.is_dir()]
    dataset_images = []
    dataset_annotations = []

    img_idx = 0
    ann_idx = 0

    for geo_location in tqdm(sample_dir, ascii=True, unit="dir"):
        img_paths = [
            i
            for i in list(geo_location.iterdir())
            if i.is_file() and i.suffix == ".jpg"
        ]
        for img_path in tqdm(img_paths, ascii=True, unit="img"):

            try:
                pil_img = Image.open(img_path)
                ret_val = get_annotation(img_path)
                if ret_val is None:
                    print(f"Skipping {img_path}")
                (top_left, width, height) = ret_val
                img_suffix = img_path.as_posix().split(alpr_dir.as_posix())[1][1:]
                img = {
                    "license": 4,
                    "file_name": f"{img_suffix}",
                    "coco_url": "TBD",
                    "height": pil_img.height,
                    "width": pil_img.width,
                    "date_captured": "UNK",
                    "flickr_url": "UNK",
                    "id": img_idx,
                }
                dataset_images.append(img)

                box_ann = {
                    "area": width * height,
                    "bbox": top_left + [width, height],
                    "category_id": 1,
                    "id": ann_idx,
                    "image_id": img_idx,
                    "iscrowd": 0,
                    "segmentation": "UNK",
                }
                dataset_annotations.append(box_ann)
                ann_idx += 1
                img_idx += 1

            except Exception as err:
                print(f"Skipping {img_path} {err}")

    annotation_data["images"] = dataset_images
    print(f"Done adding {len(dataset_images)} images")

    annotation_data["annotations"] = dataset_annotations
    print(f"Done adding {len(dataset_annotations)} annotations")

    with json_file_path.open("w") as pfile:

        json.dump(annotation_data, pfile)

    print(f"Done writing {json_file_path}")


def normalize_coco(
    ccpd_images,
    ccpd_etxt,
    json_file_path,
):

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Normalize ALPR <—> MSCOCO for training EfficientDet"
    )
    parser.add_argument(
        "-a",
        "--alpr-dir",
        dest="alpr_dir",
        help="ALPR dataset path",
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
        args.alpr_dir,
        args.json_path,
        args.to_mscoco,
        args.to_openimages,
    )
