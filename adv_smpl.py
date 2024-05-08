import cv2
from pathlib import Path
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pyrender

from scripts.pretrained_models import (
    DETECTRON2_MODEL_PATH,
    HMR2_DEFAULT_CKPT,
)

from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hmr2


if __name__ == "__main__":
    cfg_path = (
        Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    )
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = str(DETECTRON2_MODEL_PATH)
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    # detector.model.to('cuda:1')
    for img_path in tqdm(reference_img_paths, desc="Processing Reference Images:"):
        img_cv2 = cv2.imread(
            str(os.path.join(args.reference_imgs_folder, "images", img_path))
        )

        renderer.renderer.delete()
        renderer.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_cv2.shape[:2][::-1][0],
            viewport_height=img_cv2.shape[:2][::-1][1],
            point_size=1.0,
        )
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        dataloader = load_image(img_cv2, detector)

        for batch in dataloader:
            batch = recursive_to(batch, args.device)
            results_dict_for_rendering, misc_args = predict_smpl(batch, model, model_cfg, args.figure_scale)
            
            rendering_results = renderer.render_all_multiple(
                results_dict_for_rendering["verts"], 
                cam_t=results_dict_for_rendering["cam_t"], 
                render_res=results_dict_for_rendering["render_res"], **misc_args
            )
            # Overlay image
            valid_mask = rendering_results["Image"][:, :, -1][:, :, np.newaxis]
            cam_view = (
                valid_mask * rendering_results["Image"][:, :, [2, 1, 0]]
                + (1 - valid_mask) * img_cv2.astype(np.float32)[:, :, ::-1] / 255
            )
