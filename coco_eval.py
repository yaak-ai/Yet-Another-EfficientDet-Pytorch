# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import argparse
import torch
import yaml
from random import shuffle
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string

ap = argparse.ArgumentParser()
ap.add_argument(
    "-d", "--dataset", type=str, default="dataset", help="dataset root path"
)
ap.add_argument(
    "-p",
    "--project",
    type=str,
    default="coco",
    help="project file that contains parameters",
)
ap.add_argument(
    "-c", "--compound_coef", type=int, default=0, help="coefficients of efficientdet"
)
ap.add_argument("-w", "--weights", type=str, default=None, help="/path/to/weights")
ap.add_argument(
    "--threshold",
    type=float,
    default=0.2,
    help="Threshold, don't change it if not for testing purposes",
)
ap.add_argument(
    "--nms_threshold",
    type=float,
    default=0.5,
    help="nms threshold, don't change it if not for testing purposes",
)
ap.add_argument("--cuda", type=boolean_string, default=False)
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--float16", type=boolean_string, default=False)
ap.add_argument("--max-images", type=int, default=10000)
ap.add_argument(
    "--override",
    type=boolean_string,
    default=True,
    help="override previous bbox results file if exists",
)
args = ap.parse_args()

dataset = args.dataset
compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
threshold = args.threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = (
    f"weights/efficientdet-d{compound_coef}.pth"
    if args.weights is None
    else args.weights
)
max_images = args.max_images

print(
    f"running coco-style evaluation on project {project_name}, weights {weights_path}..."
)

params = yaml.safe_load(open(f"projects/{project_name}.yml"))
obj_list = params["obj_list"]

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    pbar = tqdm(image_ids, ascii=True, unit="samples")

    for image_id in pbar:
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info["file_name"]

        ori_imgs, framed_imgs, framed_metas = preprocess(
            image_path,
            max_size=input_sizes[compound_coef],
            mean=params["mean"],
            std=params["std"],
        )
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(
            x,
            anchors,
            regression,
            classification,
            regressBoxes,
            clipBoxes,
            threshold,
            nms_threshold,
        )

        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds["scores"]
        class_ids = preds["class_ids"]
        rois = preds["rois"]

        # if 0 in preds["class_ids"]:
        #
        #     import pdb
        #
        #     pdb.set_trace()

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    "image_id": image_id,
                    "category_id": label + 1,
                    "score": float(score),
                    "bbox": box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception(
            "the model does not provide any valid output, check model architecture and the data input"
        )

    # write output
    filepath = f"{set_name}_bbox_results.json"
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, "w"), indent=4)


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print("BBox")
    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    SET_NAME = params["val_set"]
    VAL_GT = f'{dataset}/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'{dataset}/{params["project_name"]}/{SET_NAME}/'
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()
    shuffle(image_ids)
    image_ids = image_ids[:max_images]

    if override_prev_results or not os.path.exists(f"{SET_NAME}_bbox_results.json"):
        model = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=len(obj_list),
            ratios=eval(params["anchors_ratios"]),
            scales=eval(params["anchors_scales"]),
        )
        model.load_state_dict(
            torch.load(weights_path, map_location=torch.device("cpu"))
        )
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model.cuda(gpu)

            if use_float16:
                model.half()

        evaluate_coco(
            VAL_IMGS, SET_NAME, image_ids, coco_gt, model, threshold=threshold
        )

    _eval(coco_gt, image_ids, f"{SET_NAME}_bbox_results.json")
