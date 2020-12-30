import argparse
from queue import Queue
from threading import Thread
from collections import namedtuple
from pathlib import Path

from tqdm import tqdm
from torch.backends import cudnn

import torch
import cv2
import numpy as np
from skvideo.io import ffprobe, vreader, FFmpegWriter
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes

from utils.utils import invert_affine, postprocess, preprocess_video, get_metas
from utils import drawing

METADATA = namedtuple("metadata", "codec fps nb_frames width height")

# H265 vcodec
FFMPEG_VCODEC = "libx265"
# H265 encode speed
FFMPEG_PRESET = "ultrafast"
# Video tag for quicktime player
FFMPEG_VTAG = "hvc1"

cudnn.fastest = True
cudnn.benchmark = True

force_input_size = None  # set None to use default size


IMAGENET_DEFAULT_MEAN = torch.tensor(
    np.array((0.485, 0.456, 0.406)) * 255, dtype=torch.float32
)
IMAGENET_DEFAULT_STD = torch.tensor(
    np.array((0.229, 0.224, 0.225)) * 255, dtype=torch.float32
)


def preprocess_yaak(frame, size):

    """
    Model specfic frame pre-processing
    1. Tesor from numpy array
    2. Mean subtraction
    3. HWC -> CHW
    4. CHN -> NCHW
    """

    # Throwing all pre-precessing onto device if possible
    # import time
    # start = time.time()
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR) + 0.0
    frame = torch.from_numpy(frame)
    frame -= IMAGENET_DEFAULT_MEAN
    frame /= IMAGENET_DEFAULT_STD
    frame = frame.permute([2, 0, 1])
    # print(f"preprocess - {time.time() - start}s")

    return frame.unsqueeze(0)


def seek_fn(
    video_file_path, queue_frames, queue_tensor, frame_count, batch_size, input_size
):
    """
    Iterator over the video frames
    Seek and you shall find it — Harsi, Circa 2020
    """

    # Opena a video reader
    print(f"Opened file {video_file_path}")
    src_reader = vreader(video_file_path.as_posix())

    pbar = tqdm(src_reader, total=frame_count, ascii=True, unit="frames")

    tensor_batch = []
    frame_batch = []
    for frame in pbar:
        if len(frame_batch) < batch_size:
            frame_batch.append(frame.copy())
            tensor_batch.append(preprocess_yaak(frame, input_size))
            continue
        queue_frames.put(frame_batch)
        queue_tensor.put(torch.cat(tensor_batch, dim=0).float().cuda())
        frame_batch, tensor_batch = [frame.copy()], [preprocess_yaak(frame, input_size)]

    # last batch
    queue_frames.put(frame_batch)
    queue_tensor.put(torch.cat(tensor_batch, dim=0).float().cuda())
    #
    src_reader.close()
    # TODO : Harsimrat — gehacked here lolz
    queue_frames.put(None)
    queue_tensor.put(None)


def get_metadata(video_file_path):

    metadata = ffprobe(video_file_path.as_posix())
    codec = metadata["video"]["@codec_name"]
    # Get a frame count H265 videos don't have "nb_frames" key in ffprobe
    nb_frames = int(metadata["video"]["@nb_frames"]) if codec == "h264" else -1
    fps = metadata["video"]["@r_frame_rate"].split("/")[0]
    height = int(metadata["video"]["@height"])
    width = int(metadata["video"]["@width"])

    m = METADATA(codec=codec, fps=fps, nb_frames=nb_frames, width=width, height=height)

    return m


def preprocess(frame, input_size):

    """
    Model specfic frame pre-processing
    1. Tesor from numpy array
    2. Mean subtraction
    3. HWC -> CHW
    4. CHN -> NCHW
    """

    frame = frame[:, :, ::-1]
    _, framed_imgs, meta = preprocess_video(frame, max_size=input_size)

    print(f"{len(framed_imgs)}")

    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)

    return x.permute(0, 3, 1, 2)


def inference_fn(model, model_frame_queue, detections_queue):

    """
    Run inference after fetching image tensor from model_frame_queue
    write back detections_queue. Frame is expected to be pre-processed
    """

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    with torch.no_grad():

        while True:
            frame = model_frame_queue.get()
            if frame is None:
                break

            import time

            start = time.time()
            features, regression, classification, anchors = model(frame)

            out = postprocess(
                frame,
                anchors,
                regression,
                classification,
                regressBoxes,
                clipBoxes,
                0.2,
                0.2,
            )
            print(f"inference - {time.time() - start}s")
            detections_queue.put(out)
    # TODO : Harsi gehacked here again for eof sig
    detections_queue.put(None)


def redact_fn(
    queue_frame, queue_detection, video_file_path, fps, threshold, input_size
):

    """
    Fetches image from frame_queue, detections from detections_queue
    Blurs/Draws prediction and writes on stream
    """

    frame_count = 0
    obj_count = 0
    print(f"Opening video writer for {video_file_path}")

    class_names = ["Vehicle registration plate", "Human face"]

    dst_writer = FFmpegWriter(
        video_file_path.as_posix(),
        outputdict={
            "-vcodec": FFMPEG_VCODEC,
            "-r": fps,
            "-preset": FFMPEG_PRESET,
            "-vtag": FFMPEG_VTAG,
        },
    )

    metas = get_metas((1080, 1920, 3), input_size, input_size)

    while True:
        frames = queue_frame.get()
        objs = queue_detection.get()
        if frames is None:
            break

        obj_metas = [metas] * len(objs)

        objs = invert_affine(obj_metas, objs)

        for frame, obj in zip(frames, objs):
            # obj = [b for b in obj if float(b["scores"]) > threshold[int(b["class_id"])]]
            class_ids = [c for c in obj["class_ids"]]
            boxes = [b for b in obj["rois"]]
            scores = [s for s in obj["scores"]]
            classes = [class_names[int(b)] for b in class_ids]
            detections = list(zip(classes, scores, boxes))
            frame = drawing.redact_regions(frame, detections)
            frame = drawing.draw_rectangle(frame, boxes)
            dst_writer.writeFrame(frame)
            frame_count += 1
            obj_count += len(detections)
            # [x_min, y_min, x_max, y_max, score, class]

    dst_writer.close()
    print(f"Found {obj_count} redacted objects in {frame_count} frames")


def redact(
    compound_coef,
    weights_path,
    source_vid_path,
    dst_vid_path,
    results_json,
    threshold,
    batch_size=1,
):

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = (
        input_sizes[compound_coef] if force_input_size is None else force_input_size
    )

    obj_list = ["Vehicle registration plate", "Human face"]

    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    # anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    anchor_scales = [0.3, 0.5, 0.8, 2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    model = EfficientDetBackbone(
        compound_coef=compound_coef,
        num_classes=len(obj_list),
        ratios=anchor_ratios,
        scales=anchor_scales,
    )
    model.load_state_dict(torch.load(weights_path))
    model.requires_grad_(False)
    model.eval()
    model.cuda()

    param_count = sum([m.numel() for m in model.parameters()])
    print(f"Model {EfficientDetBackbone} created, param count: {param_count}")

    queue_frames = Queue()
    queue_tensor = Queue(maxsize=1)
    queue_detection = Queue(maxsize=1)

    metadata = get_metadata(Path(source_vid_path))

    print(
        f"{source_vid_path} {metadata.codec} {metadata.fps} fps {metadata.nb_frames} frames"
    )

    if metadata.nb_frames == 0:
        print(f"Empty file ? {source_vid_path}")
        return

    print(f"Redacted video WIDTHxHEIGHT {metadata.width}x{metadata.height}")

    t0 = Thread(
        target=seek_fn,
        args=(
            Path(source_vid_path),
            queue_frames,
            queue_tensor,
            metadata.nb_frames,
            batch_size,
            input_size,
        ),
    )
    t0.start()
    t1 = Thread(
        target=inference_fn,
        args=(model, queue_tensor, queue_detection),
    )
    t1.start()
    t2 = Thread(
        target=redact_fn,
        args=(
            queue_frames,
            queue_detection,
            Path(dst_vid_path),
            metadata.fps,
            threshold,
            input_size,
        ),
    )
    t2.start()
    t2.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run PII model on Yaak Drive Data")
    parser.add_argument(
        "-c", "--coff", dest="coff", help="EfficientDet model coff", type=int
    )
    parser.add_argument("-w", "--weights", dest="weights", help="PII Model config")
    parser.add_argument("-s", "--source_vid", dest="video", help="Video path")
    parser.add_argument("-r", "--results-json", dest="json", help="Json path")
    parser.add_argument("-d", "--dest_vid", dest="video_out", help="Output Video path")
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        help="Batch size",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-t",
        "--thresh",
        dest="threshold",
        default=[0.1, 0.2],
        type=float,
        nargs=2,
        help="Model Threshold",
    )

    args = parser.parse_args()

    redact(
        args.coff,
        args.weights,
        args.video,
        args.video_out,
        args.json,
        args.threshold,
        args.batch_size,
    )
