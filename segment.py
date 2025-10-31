# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng
# Modified by Yagi: folder input + ID/contour overlay

import argparse
import glob
import multiprocessing as mp
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from demo.predictor import VisualizationDemo

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using device:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())

WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="Path to image or directory (supports folder input).",
    )
    parser.add_argument(#今コレは意味をなしてない、光の条件を変えたデータセット必要だな
        "--output",
        help="Output file or directory for visualizations.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        return os.path.isfile(filename)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

   
    if args.input:
        input_path = os.path.expanduser(args.input[0])
        if os.path.isdir(input_path):
            args.input = sorted(
                glob.glob(os.path.join(input_path, "*.png")) +
                glob.glob(os.path.join(input_path, "*.jpg"))
            )
            print(f" Found {len(args.input)} images in folder: {input_path}")
        else:
            args.input = glob.glob(input_path)
        assert args.input, f"No images found at: {input_path}"

        # 出力ディレクトリ準備
        os.makedirs("outputdata", exist_ok=True)

        for i, path in enumerate(tqdm.tqdm(args.input)):
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)

            if "instances" in predictions:
                instances = predictions["instances"].to("cpu")
                masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
                vis_img = visualized_output.get_image()  # RGB ndarray

                if masks is not None:
                    for mid, mask in enumerate(masks):
                        # 面積フィルタ 
                        min_area_pixels = 50
                        min_area_ratio = 0.001
                        area_pixels = int(mask.sum())
                        h, w = mask.shape
                        area_ratio = area_pixels / (h * w)
                        if area_pixels < min_area_pixels or area_ratio < min_area_ratio:
                            mask_bool = mask.astype(bool)
                            orig_rgb = img[:, :, ::-1]
                            vis_img[mask_bool] = orig_rgb[mask_bool]
                            continue

                        # IDと輪郭描画 
                        ys, xs = np.where(mask)
                        if xs.size > 0 and ys.size > 0:
                            cx, cy = int(xs.mean()), int(ys.mean())
                            cv2.putText(vis_img, f"{mid}", (cx-10, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            contours, _ = cv2.findContours(mask.astype(np.uint8),
                                                           cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(vis_img, contours, -1, (255, 255, 255), 1)

                # --- 保存 ---
                vis_img_bgr = vis_img[:, :, ::-1]
                out_name = f"outputdata/frame_{i:04d}.png"
                cv2.imwrite(out_name, vis_img_bgr)
                print(f"Saved: {out_name}")

            logger.info(
                f"{path}: detected {len(predictions['instances']) if 'instances' in predictions else 0} instances "
                f"in {time.time() - start_time:.2f}s"
            )


    elif args.webcam:
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break
        cam.release()
        cv2.destroyAllWindows()

    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        output_name = os.path.splitext(basename)[0] + file_ext
        output_path = os.path.join("outputdata", output_name)
        os.makedirs("outputdata", exist_ok=True)
        writer = cv2.VideoWriter(
            filename=output_path,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(fps),
            frameSize=(width, height),
            isColor=True,
        )

        for vis_frame in tqdm.tqdm(demo.run_on_video(video)):
            writer.write(vis_frame)
        video.release()
        writer.release()
        print(f"🎥 Saved video: {output_path}")
