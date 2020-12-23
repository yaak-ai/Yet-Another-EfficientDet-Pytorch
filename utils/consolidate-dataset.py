import argparse
from collections import namedtuple
from pathlib import Path

import json
from tqdm import tqdm


def consolidate(
    json_list,
    image_list,
    json_file_path,
):
    annotation_data = {
        "annotations": None,
        "categories": None,
        "images": None,
        "info": None,
        "licenses": "yaak.ai",
    }

    categories = []

    classes = [[0, "Vehicle registration plate", "LP"], [1, "Human Face", "HF"]]

    for idx, name, machine_name in tqdm(classes, ascii=True, total=len(classes)):

        d = {
            "supercategory": "PII",
            "id": idx + 1,
            "name": name,
            "machine_name": machine_name,
        }
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

    dataset_images = []
    dataset_annotations = []

    img_idx = 0
    ann_idx = 0

    with open(json_list) as pfile:
        jsons = pfile.readlines()

    with open(image_list) as pfile:
        images = pfile.readlines()

    jsons = [j.strip() for j in jsons]
    images = [i.strip() for i in images]

    for filepath, img_prefix in tqdm(zip(jsons, images), ascii=True, unit="json"):
        with open(filepath) as pfile:
            json_data = json.load(pfile)
        images = json_data["images"]
        annotations = json_data["annotations"]

        for image in tqdm(images, ascii=True, unit="img", leave=False):
            image["id"] += img_idx
            image["file_name"] = f"{img_prefix}/{image['file_name']}"
            dataset_images.append(image)
        for annotation in tqdm(annotations, ascii=True, unit="ann", leave=False):
            annotation["image_id"] += img_idx
            annotation["id"] += ann_idx
            dataset_annotations.append(annotation)

        img_idx += len(images)
        ann_idx += len(annotations)

    annotation_data["images"] = dataset_images
    print(f"Done adding {len(dataset_images)} images")

    annotation_data["annotations"] = dataset_annotations
    print(f"Done adding {len(dataset_annotations)} annotations")

    with json_file_path.open("w") as pfile:

        json.dump(annotation_data, pfile)

    print(f"Done writing {json_file_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Consolidate datasets for MSCOCO training")
    parser.add_argument(
        "-l",
        "--json-list",
        dest="json_list",
        help="JSON list path",
        type=Path,
    )
    parser.add_argument(
        "-i",
        "--image-list",
        dest="image_list",
        help="JSON list path",
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--destination",
        type=Path,
        dest="json_path",
        help="Json Path",
    )
    args = parser.parse_args()

    consolidate(
        args.json_list,
        args.image_list,
        args.json_path,
    )
