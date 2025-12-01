#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import yaml
import numpy as np
import random
from collections import deque
import sys
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.join(CURRENT_DIR, '..', 'instance_segmentation_ros/scripts')
sys.path.append(SCRIPT_DIR)

from trackking import match_instances
from light_tracking import iou_match_instances
from util import load_data, multi_detect

# --- Detectron2 / Mask2Former 関連 ---
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config


DATA_DIR = os.path.expanduser(
    "~/catkin_ws/yagidata/tracking_testdata/mornig_sunny_test_data_20251101"
)
SAVE_DIR = os.path.join(DATA_DIR, "Mask2former/segmented")
SAVE_DIRcolor = os.path.join(DATA_DIR, "Mask2former/segmented_color")
SAVE_DIR_mask = os.path.join(DATA_DIR, "Mask2former/segmented_mask")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIRcolor, exist_ok=True)
os.makedirs(SAVE_DIR_mask, exist_ok=True)


class OfflineSegmentationTracker:
    def __init__(self, config_path="/home/ryotayagi/catkin_ws/src/instance_segmentation_ros/config/realsense.yaml",
                 history_len=6):

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.imgsz = cfg.get("imgsz", 640)
        self.conf = cfg.get("conf_threshold", 0.4)
        self.mask_th = cfg.get("mask_threshold", 0.4)

        mask2_cfg_file = cfg.get("mask2former_config",
                                 "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
        mask2_weights = cfg.get("mask2former_weights", "weights/model_final.pth")

        dcfg = get_cfg()
        add_deeplab_config(dcfg)
        add_maskformer2_config(dcfg)
        dcfg.merge_from_file(mask2_cfg_file)

        dcfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf
        dcfg.MODEL.WEIGHTS = mask2_weights
        device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        dcfg.MODEL.DEVICE = device

        self.predictor = DefaultPredictor(dcfg)

        self.next_id = 1
        self.color = {}
        self.history_len = history_len
        self.history = deque(maxlen=history_len)
        self.id_miss = {}

    def get_color(self, inst_id):
        if inst_id not in self.color:
            self.color[inst_id] = [random.randint(0, 255) for _ in range(3)]
        return self.color[inst_id]


    def visualize_save_masks(self, img, outputs, save_dir, frame_idx):

        from detectron2.structures import Instances

        if not isinstance(outputs, dict) or "instances" not in outputs:
            print(f"[Frame {frame_idx}] visualize: No instances in output dict.")
            return

        instances = outputs["instances"]
        if not isinstance(instances, Instances):
            print(f"[Frame {frame_idx}] visualize: instances is not Instances.")
            return

        if not instances.has("pred_masks"):
            print(f"[Frame {frame_idx}] visualize: No pred_masks field.")
            return

        masks = instances.pred_masks  # BoolTensor(N,H,W)

        if len(masks) == 0:
            print(f"[Frame {frame_idx}] visualize: mask empty.")
            return

        vis_masks = masks.cpu().numpy()  # (N, H, W)

        yolo_vis = img.copy()
        H_img, W_img = yolo_vis.shape[:2]

        for idx, mask in enumerate(vis_masks):
            mask_bin = (mask > 0.5).astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_bin, (W_img, H_img),
                                      interpolation=cv2.INTER_NEAREST)

            hue = int(180 * (idx / max(1, len(vis_masks))))
            hsv_color = np.uint8([[[hue, 200, 255]]])
            bgr_color = tuple(int(c) for c in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0])

            colored_layer = np.zeros_like(yolo_vis, dtype=np.uint8)
            for c in range(3):
                colored_layer[:, :, c] = np.where(mask_resized == 255, bgr_color[c], 0)

            alpha_mask = (mask_resized.astype(np.float32) / 255.0) * 0.6
            alpha_mask_3 = np.repeat(alpha_mask[:, :, None], 3, axis=2)

            yolo_vis = (
                yolo_vis.astype(np.float32) * (1.0 - alpha_mask_3)
                + colored_layer.astype(np.float32) * alpha_mask_3
            ).astype(np.uint8)

            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(yolo_vis, contours, -1, bgr_color, 2)

        save_path = os.path.join(save_dir, f"mask_vis_{frame_idx:04d}.png")
        cv2.imwrite(save_path, yolo_vis)
        print(f"[Frame {frame_idx}] Saved visualization → {save_path}")


    def run(self):
        rgb_files, poses, F_list = load_data(self, data_dir=DATA_DIR)
        print(f"Processing {len(rgb_files)} frames...")

        for i, color_path in enumerate(rgb_files):
            img = cv2.imread(color_path)
            if img is None:
                print(f"[WARN] Failed to read {color_path}")
                continue
            H, W = img.shape[:2]

            outputs = self.predictor(img)######################################################################インスタンスIDが膨大になる原因修正
            instances = outputs["instances"].to("cpu")

            # ---- visualization ----
            try:
                self.visualize_save_masks(img, outputs, SAVE_DIR_mask, i)
            except Exception as e:
                print(f"[WARN] visualize_save_masks failed: {e}")

            # ---- pred_masks → numpy list ----
            masks = []
            if instances.has("pred_masks"):
                pred_masks = instances.pred_masks  # BoolTensor(N,H,W)
                for m in pred_masks:
                    m_np = m.numpy().astype(np.uint8) * 255
                    if m_np.shape != (H, W):
                        m_np = cv2.resize(m_np, (W, H), interpolation=cv2.INTER_NEAREST)
                    masks.append(m_np)

            if len(masks) == 0:
                print(f"[Frame {i+1}] No detections.")
                for id_ in self.id_miss.keys():
                    self.id_miss[id_] += 1
                continue

            # ---- ID assignment ----
            ids = [-1] * len(masks)

            # 過去フレームとのマッチング
            for past_masks, past_ids in reversed(self.history):
                matches = match_instances(
                    past_masks, masks,
                    F=F_list[i], alpha=0.3,
                    iou_threshold=0.25, tol=2.0, device="cuda"
                )

                for pid, cid in matches.items():
                    if cid >= 0 and pid < len(past_ids):
                        if ids[cid] == -1:
                            ids[cid] = past_ids[pid]
                            self.id_miss[past_ids[pid]] = 0

            # 新規ID
            for k in range(len(masks)):
                if ids[k] == -1:
                    ids[k] = self.next_id
                    self.next_id += 1
                    self.id_miss[ids[k]] = 0

            # ---- ID map ----
            id_map = np.zeros((H, W), dtype=np.uint8)
            for mask, inst_id in zip(masks, ids):
                id_map[mask > 0] = inst_id + 1

            # ---- color visualization ----
            color_img = np.zeros((H, W, 3), dtype=np.uint8)
            for inst_id in np.unique(id_map):
                if inst_id == 0:
                    continue
                col = self.get_color(inst_id)
                color_img[id_map == inst_id] = col

                ys, xs = np.where(id_map == inst_id)
                if xs.size > 0 and ys.size > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    cv2.putText(color_img, f"ID {inst_id-1}", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)

            save_color = os.path.join(SAVE_DIRcolor, f"idmap_color_{i:04d}.png")
            save_idmap = os.path.join(SAVE_DIR, f"idmap_{i:04d}.png")

            cv2.imwrite(save_color, color_img)
            cv2.imwrite(save_idmap, id_map)

            print(f"[Frame {i+1}] Saved {save_color} (instances={len(masks)}, next_id={self.next_id})")

            # ---- update history ----
            self.history.append((masks, ids))

            for id_ in list(self.id_miss.keys()):
                if all(id_ not in frame_ids for _, frame_ids in self.history):
                    self.id_miss[id_] += 1

            remove_ids = [id_ for id_, miss in self.id_miss.items()
                          if miss > self.history_len]
            for rid in remove_ids:
                del self.id_miss[rid]

        print(f"Done. {self.next_id} unique IDs were used.")


if __name__ == "__main__":
    tracker = OfflineSegmentationTracker(history_len=5)
    tracker.run()
