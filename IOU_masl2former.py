import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, "Mask2Former")

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from mask2former import add_maskformer2_config
from pycocotools.coco import COCO

def get_image_id_from_filename(coco, filename):
    for img in coco.dataset["images"]:
        if img["file_name"] == filename:
            return img["id"]
    print(f"[ERROR] file_name '{filename}' が JSON に存在しません")
    return None


def load_coco_polygon_masks(json_path, image_id):
    coco = COCO(json_path)
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))

    img_info = coco.loadImgs([image_id])[0]
    h, w = img_info["height"], img_info["width"]

    masks = []
    for ann in anns:
        seg = ann["segmentation"]

        mask = np.zeros((h, w), dtype=np.uint8)

        for poly in seg:
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

        masks.append((ann["category_id"], mask))

    return masks

def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    return inter / union if union > 0 else 0.0



def visualize_instance_masks(image, masks, alpha=0.5, seed=0):
    np.random.seed(seed)
    overlay = np.zeros_like(image)

    for _, mask in masks:
        color = np.random.randint(0, 255, size=3)
        colored = np.zeros_like(image)

        for c in range(3):
            colored[:, :, c] = np.where(mask > 0, color[c], 0)

        overlay = cv2.addWeighted(overlay, 1.0, colored, 1.0, 0)

    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)



def load_mask2former(cfg_path, weights_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(cfg_path)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False

    cfg.MODEL.DEVICE = "cuda"
    return DefaultPredictor(cfg)



def run_mask2former(predictor, image):
    outputs = predictor(image)
    inst = outputs["instances"].to("cpu")

    pred_masks = []
    for mask, cls in zip(inst.pred_masks.numpy(), inst.pred_classes.numpy()):
        mask = (mask.astype(np.uint8)) * 255
        pred_masks.append((int(cls), mask))

    return pred_masks



def average_iou(gt_masks, pred_masks):
    if len(pred_masks) == 0:
        return 0.0

    total = 0.0
    for cls_gt, mask_gt in gt_masks:
        best = 0.0
        for cls_pr, mask_pr in pred_masks:
            best = max(best, compute_iou(mask_gt, mask_pr))
        total += best

    return total / len(gt_masks)

if __name__ == "__main__":

    img_path = Path("datasets/my_dataset/images/pose_00_20251008_102948_758.png")
    json_path = Path("datasets/my_dataset/annotations/instances_default_poly.json")

    m2f_cfg = "/home/ryotayagi/catkin_ws/src/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
    m2f_weight = "/home/ryotayagi/catkin_ws/src/Mask2Former/weights/model_final.pth"

    # 画像読み込み
    image = cv2.imread(str(img_path))
    if image is None:
        print("[ERROR] 画像が読み込めません:", img_path)
        exit()

    # COCO 読み込み
    coco = COCO(json_path)
    filename = img_path.name

    # 画像ファイル名から image_id を取得
    image_id = get_image_id_from_filename(coco, filename)
    if image_id is None:
        exit()

    # GT マスクを取得
    gt_masks = load_coco_polygon_masks(json_path, image_id)

    # Mask2Former 推論
    predictor = load_mask2former(m2f_cfg, m2f_weight)
    pred_masks = run_mask2former(predictor, image)

    # IoU 計算
    iou_result = average_iou(gt_masks, pred_masks)

    print("\n============== IoU RESULT ==============")
    print(f"Mask2Former Average IoU = {iou_result:.3f}")
    print("=========================================\n")

    # 可視化表示
    vis_gt = visualize_instance_masks(image, gt_masks, alpha=0.5, seed=1)
    vis_pr = visualize_instance_masks(image, pred_masks, alpha=0.5, seed=3)

    cv2.imshow("GT", vis_gt)
    cv2.imshow("Mask2Former", vis_pr)
    # cv2.imwrite("output_data/vis_gt.png", vis_gt)
    # cv2.imwrite("output_data/vis_mask2former.png", vis_pr)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
