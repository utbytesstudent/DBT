# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects import point_rend  # Optional, for enhanced visualization
from detectron2.utils.visualizer import Visualizer, ColorMode

# constants
WINDOW_NAME = "Skeleton Tracking"

def setup_cfg(args):
    # Load config for keypoint detection
    cfg = get_cfg()
    # Keypoint R-CNN model for human pose estimation
    cfg.merge_from_file("configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Skeleton Tracking using Detectron2")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--output", help="Path to save output video with skeleton tracking.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum score for instance predictions to be shown.")
    return parser

def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger()
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    # Open video file
    video = cv2.VideoCapture(args.video_input)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.output:
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        output_file = cv2.VideoWriter(args.output, codec, frames_per_second, (width, height))

    assert os.path.isfile(args.video_input)

    for _ in tqdm.tqdm(range(num_frames)):
        ret, frame = video.read()
        if not ret:
            break

        # Run Detectron2's keypoint predictor on the frame
        outputs = predictor(frame)

        # Visualize the keypoints and skeleton
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis_frame = v.get_image()[:, :, ::-1]

        # Show the skeleton tracking in a window
        cv2.imshow(WINDOW_NAME, vis_frame)

        # Write to output video file if specified
        if args.output:
            output_file.write(vis_frame)

        # Press 'Esc' to stop
        if cv2.waitKey(1) == 27:
            break

    video.release()
    if args.output:
        output_file.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
