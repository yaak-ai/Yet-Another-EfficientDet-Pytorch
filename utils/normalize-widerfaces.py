# FROM HERE — https://gist.github.com/AruniRC/8913ce12b0903d1e68aeff3fc7a1dc42

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import sys
from PIL import Image
from pathlib import Path


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)
add_path(this_dir)
# print(this_dir)
add_path(os.path.join(this_dir, "..", ".."))


# INFO = {
#     "description": "WIDER Face Dataset",
#     "url": "http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/",
#     "version": "0.1.0",
#     "year": 2018,
#     "contributor": "umass vision",
#     "date_created": datetime.datetime.utcnow().isoformat(' ')
# }

# LICENSES = [
#     {
#         "id": 1,
#         "name": "placeholder",
#         "url": "placeholder"
#     }
# ]

# CATEGORIES = [
#     {
#         'id': 2,
#         'name': 'face',
#         'supercategory': 'face',
#     },
# ]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument("--dataset", help="wider", default="wider", type=str)
    parser.add_argument(
        "--json-path",
        dest="json_path",
        help="Output JSON file",
        type=Path,
    )
    parser.add_argument(
        "--img-dir",
        dest="img_dir",
        help="root directory for loading dataset images",
        default="data/WIDER",
        type=Path,
    )
    parser.add_argument(
        "--annotfile",
        dest="annotation_file",
        help="directly specify the annotations file",
        default="",
        type=Path,
    )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


# -----------------------------------------------------------------------------------------
def parse_wider_gt(dets_file_name, isEllipse=False):
    # -----------------------------------------------------------------------------------------
    """
    Parse the FDDB-format detection output file:
      - first line is image file name
      - second line is an integer, for `n` detections in that image
      - next `n` lines are detection coordinates
      - again, next line is image file name
      - detections are [x y width height score]

    Returns a dict: {'img_filename': detections as a list of arrays}
    """
    fid = dets_file_name.open("r")

    # Parsing the FDDB-format detection output txt file
    img_flag = True
    numdet_flag = False
    start_det_count = False
    det_count = 0
    numdet = -1

    det_dict = {}
    img_file = ""

    for line in fid:
        line = line.strip()

        if img_flag:
            # Image filename
            img_flag = False
            numdet_flag = True
            # print 'Img file: ' + line
            img_file = line
            det_dict[img_file] = []  # init detections list for image
            continue

        if numdet_flag:
            # next line after image filename: number of detections
            numdet = int(line)
            numdet_flag = False
            if numdet > 0:
                start_det_count = True  # start counting detections
                det_count = 0
            else:
                # no detections in this image
                img_flag = True  # next line is another image file
                numdet = -1

            # print 'num det: ' + line
            continue

        if start_det_count:
            # after numdet, lines are detections
            detection = [float(x) for x in line.split()]  # split on whitespace
            det_dict[img_file].append(detection)
            # print 'Detection: %s' % line
            det_count += 1

        if det_count == numdet:
            start_det_count = False
            det_count = 0
            img_flag = True  # next line is image file
            numdet_flag = False
            numdet = -1

    return det_dict


def convert_wider_annots(dataset_name, annotation_file, img_dir, json_file):
    """Convert from WIDER FDDB-style format to COCO bounding box"""

    img_id = 0
    ann_id = 0
    cat_id = 2

    print("Starting %s" % dataset_name)
    ann_dict = {}
    categories = [{"id": cat_id, "name": "face"}]
    images = []
    annotations = []
    ann_file = annotation_file
    wider_annot_dict = parse_wider_gt(ann_file)  # [im-file] = [[x,y,w,h], ...]

    for filename in wider_annot_dict.keys():
        if len(images) % 50 == 0:
            print(
                "Processed %s images, %s annotations" % (len(images), len(annotations))
            )

        image = {}
        image["id"] = img_id
        img_id += 1
        im = Image.open(os.path.join(img_dir, filename))
        image["width"] = im.height
        image["height"] = im.width
        image["file_name"] = filename
        images.append(image)

        for gt_bbox in wider_annot_dict[filename]:
            ann = {}
            ann["id"] = ann_id
            ann_id += 1
            ann["image_id"] = image["id"]
            ann["segmentation"] = []
            ann["category_id"] = cat_id  # 1:"face" for WIDER
            ann["iscrowd"] = 0
            ann["area"] = gt_bbox[2] * gt_bbox[3]
            ann["bbox"] = gt_bbox[:4]
            annotations.append(ann)

    ann_dict["images"] = images
    ann_dict["categories"] = categories
    ann_dict["annotations"] = annotations
    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
    print("Num annotations: %s" % len(annotations))
    with json_file.open("w", encoding="utf8") as outfile:
        outfile.write(json.dumps(ann_dict))


def convert_cs6_annots(ann_file, im_dir, out_dir, data_set="CS6-subset"):
    """Convert from WIDER FDDB-style format to COCO bounding box"""

    if data_set == "CS6-subset":
        json_name = "cs6-subset_face_train_annot_coco_style.json"
        # ann_file = os.path.join(data_dir, 'wider_face_train_annot.txt')
    else:
        raise NotImplementedError

    img_id = 0
    ann_id = 0
    cat_id = 1

    print("Starting %s" % data_set)
    ann_dict = {}
    categories = [{"id": 1, "name": "face"}]
    images = []
    annotations = []

    wider_annot_dict = parse_wider_gt(ann_file)  # [im-file] = [[x,y,w,h], ...]

    for filename in wider_annot_dict.keys():
        if len(images) % 50 == 0:
            print(
                "Processed %s images, %s annotations" % (len(images), len(annotations))
            )

        image = {}
        image["id"] = img_id
        img_id += 1
        im = Image.open(os.path.join(im_dir, filename))
        image["width"] = im.height
        image["height"] = im.width
        image["file_name"] = filename
        images.append(image)

        for gt_bbox in wider_annot_dict[filename]:
            ann = {}
            ann["id"] = ann_id
            ann_id += 1
            ann["image_id"] = image["id"]
            ann["segmentation"] = []
            ann["category_id"] = cat_id  # 1:"face" for WIDER
            ann["iscrowd"] = 0
            ann["area"] = gt_bbox[2] * gt_bbox[3]
            ann["bbox"] = gt_bbox
            annotations.append(ann)

    ann_dict["images"] = images
    ann_dict["categories"] = categories
    ann_dict["annotations"] = annotations
    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
    print("Num annotations: %s" % len(annotations))
    with open(os.path.join(out_dir, json_name), "w", encoding="utf8") as outfile:
        outfile.write(json.dumps(ann_dict))


if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "wider":
        convert_wider_annots(
            args.dataset, args.annotation_file, args.img_dir, args.json_path
        )
    if args.dataset == "cs6-subset":
        convert_cs6_annots(
            args.annotfile, args.imdir, args.outdir, data_set="CS6-subset"
        )
    else:
        print("Dataset not supported: %s" % args.dataset)
