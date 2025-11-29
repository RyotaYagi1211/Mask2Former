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
from util import load_data, visualize_save_masks, multi_detect
# --- Detectron2 / Mask2Former 関連 ---
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
# mask2former の add 設定関数（Mask2Formerリポジトリがパスにあること）
from mask2former import add_maskformer2_config

DATA_DIR = os.path.expanduser("~/catkin_ws/yagidata/tracking_testdata/mornig_sunny_test_data_20251101")#データセットのパス
SAVE_DIR = os.path.join(DATA_DIR, "Mask2former/segmented")
SAVE_DIRcolor = os.path.join(DATA_DIR, "Mask2former/segmented_color")
SAVE_DIR_mask = os.path.join(DATA_DIR, "Mask2former/segmented_mask")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIRcolor, exist_ok=True)
os.makedirs(SAVE_DIR_mask, exist_ok=True)

class OfflineSegmentationTracker:
    def __init__(self, config_path="/home/ryotayagi/catkin_ws/src/instance_segmentation_ros/config/realsense.yaml", history_len=6):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.imgsz = cfg.get("imgsz", 640)
        self.conf = cfg.get("conf_threshold", 0.4)      # ここはスコア閾値として使用
        self.mask_th = cfg.get("mask_threshold", 0.4)

        mask2_cfg_file = cfg.get("mask2former_config", "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
        mask2_weights = cfg.get("mask2former_weights", "weights/model_final.pth")  

        dcfg = get_cfg()
        add_deeplab_config(dcfg)
        add_maskformer2_config(dcfg)
        dcfg.merge_from_file(mask2_cfg_file)
        # ROI_HEADS.SCORE_THRESH_TEST はカテゴリ検出閾値に対応（モデルによっては別のキー）
        dcfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf
        dcfg.MODEL.WEIGHTS = mask2_weights
        device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        dcfg.MODEL.DEVICE = device

        self.predictor = DefaultPredictor(dcfg)

        self.next_id = 1                      # 新しいIDを連番で割り当て IDは１始まり
        self.color = {}                       # 各IDの可視化用カラー
        self.history_len = history_len        # 保持するフレーム数
        self.history = deque(maxlen=history_len)  # (masks, ids) の履歴
        self.id_miss = {}                     # id → 見失いフレーム数

    def get_color(self, inst_id):
        if inst_id not in self.color:
            self.color[inst_id] = [random.randint(0, 255) for _ in range(3)]
        return self.color[inst_id]

    def run(self):
        rgb_files, poses, F_list = load_data(self,data_dir=DATA_DIR)

        print(f"Processing {len(rgb_files)} frames...")

        for i, color_path in enumerate(rgb_files):#ここで一枚づつ処理
            img = cv2.imread(color_path)
            if img is None:
                print(f"[WARN] Failed to read {color_path}")
                continue
            H, W = img.shape[:2]


            outputs = self.predictor(img)
            results = []

            class _DummyResult:
                pass

            if "instances" in outputs and len(outputs["instances"]) > 0:
                instances = outputs["instances"].to("cpu")
                pred_masks = None
                if hasattr(instances, "pred_masks") and instances.pred_masks is not None:
                    pred_masks = instances.pred_masks  # CPU 上
                elif hasattr(instances, "pred_boxes") and instances.has("pred_boxes"):
                    pred_masks = None

                # スコアでフィルタ（scores がある場合）
                scores = instances.scores if hasattr(instances, "scores") else None

                if pred_masks is not None:
                    # pred_masks は (N, H, W) で bool/uint8/byte の tensor
                    selected_tensors = []
                    for idx in range(pred_masks.shape[0]):
                        if scores is not None:
                            score = float(scores[idx].item())
                            if score < self.conf:
                                continue
                        m = pred_masks[idx].to(torch.uint8)  # 0/1
                        # 既存コードは results[0].masks.data に torch.Tensor の配列を期待しているのでその形に揃える
                        selected_tensors.append(m)

                    if len(selected_tensors) > 0:
                        dummy = _DummyResult()
                        dummy.masks = type("m", (), {})()
                        dummy.masks.data = selected_tensors  # list of torch tensors (cpu)
                        results.append(dummy)


            ##ここでマスクの面積の閾値処理を入れても良いかも
            ##visualize save
            # visualize_save_masks(self, img, results, SAVE_DIR_mask, i)。
            try:
                visualize_save_masks(self, img, results, SAVE_DIR_mask, i)
            except Exception as e:
                print(f"[WARN] visualize_save_masks failed: {e}")

            masks = []
            if len(results) > 0 and getattr(results[0], "masks", None) is not None:
                for m in results[0].masks.data:
                    m_np = m.cpu().numpy()
                    if m_np.dtype != np.uint8:
                        m_np = (m_np > 0).astype(np.uint8) * 255
                    else:
                        if m_np.max() == 1:
                            m_np = m_np * 255
                    if (m_np.shape[0], m_np.shape[1]) != (H, W):
                        m_np = cv2.resize(m_np, (W, H), interpolation=cv2.INTER_NEAREST)
                    masks.append(m_np.astype(np.uint8))


            if len(masks) == 0:
                print(f"[Frame {i+1}] No detections (IDs remain active).")
                for id_ in self.id_miss.keys():#辞書のキーをループ
                    self.id_miss[id_] += 1
                continue

            ids = [-1] * len(masks)
            for past_masks, past_ids in reversed(self.history):
                matches = match_instances(past_masks, masks, F=F_list[i], alpha=0.3, iou_threshold=0.25, tol=2.0, device="cuda")
                for pid, cid in matches.items():
                    if cid >= 0 and pid < len(past_ids):
                        if ids[cid] == -1:
                            ids[cid] = past_ids[pid]
                            self.id_miss[past_ids[pid]] = 0  # 見つかったのでmissリセット

            for k in range(len(masks)):
                if ids[k] == -1:
                    ids[k] = self.next_id
                    self.next_id += 1
                    self.id_miss[ids[k]] = 0

            id_map = np.zeros((H, W), dtype=np.uint8)
            for mask, inst_id in zip(masks, ids):##ここでインスタンス毎のマスクをIDマップに統合してる
                id_map[mask > 0] = inst_id + 1

            color_img = np.zeros((H, W, 3), dtype=np.uint8)
            for inst_id in np.unique(id_map):
                if inst_id == 0:
                    continue
                color = self.get_color(inst_id)
                color_img[id_map == inst_id] = color
                ys, xs = np.where(id_map == inst_id)
                if xs.size > 0 and ys.size > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    cv2.putText(color_img, f"ID {inst_id-1}", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            save_idmap = os.path.join(SAVE_DIR, f"idmap_{i:04d}.png")
            save_color = os.path.join(SAVE_DIRcolor, f"idmap_color_{i:04d}.png")
            # cv2.imwrite(save_idmap, id_map)
            cv2.imwrite(save_color, color_img)
            print(f"[Frame {i+1}] Saved {save_idmap} (instances={len(masks)}, next_id={self.next_id})")

            self.history.append((masks, ids))

            for id_ in list(self.id_miss.keys()):
                if all(id_ not in frame_ids for _, frame_ids in self.history):
                    self.id_miss[id_] += 1

            # 
            remove_ids = [id_ for id_, miss in self.id_miss.items() if miss > self.history_len]
            for rid in remove_ids:
                del self.id_miss[rid]

        print(f"Done. {self.next_id} unique IDs were used.")


if __name__ == "__main__":
    tracker = OfflineSegmentationTracker(history_len=5)
    tracker.run()
