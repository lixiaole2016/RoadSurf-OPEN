import os
import numpy as np

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.kitti_eval_python.eval import do_eval
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
from lib.datasets.kitti.kitti_eval_python.rope_score import get_rope_score_eval_result


class RoadSurfDataset(KITTI_Dataset):
    """扩展 KITTI 数据集，附带 denorm 地面平面参数供 RoadSurf 使用。"""

    def __init__(self, split, cfg):
        super().__init__(split, cfg)
        self.denorm_dir = cfg.get('denorm_dir', os.path.join(self.data_dir, 'denorm'))
        os.makedirs(self.denorm_dir, exist_ok=True)

    def _load_ground_plane(self, idx):
        """读取 denorm 平面参数 (alpha, beta, gamma, d)。"""
        plane_path = os.path.join(self.denorm_dir, f'{idx:06d}.txt')
        if not os.path.exists(plane_path):
            raise FileNotFoundError(f'Ground plane file not found: {plane_path}')
        with open(plane_path, 'r') as f:
            line = f.readline()
        parts = line.replace(',', ' ').split()
        if len(parts) < 4:
            raise ValueError(f'Invalid ground plane format in {plane_path}')
        plane = np.array([float(p) for p in parts[:4]], dtype=np.float32)
        return plane

    def _get_ap3d_r40_at_iou(self, gt_annos, dt_annos, current_class, iou_thresh):
        min_overlaps = np.array([[[iou_thresh], [iou_thresh], [iou_thresh]]])
        compute_aos = any(anno['alpha'].shape[0] != 0 and anno['alpha'][0] != -10 for anno in dt_annos)
        _, _, _, _, _, _, map_3d_r40, _ = do_eval(
            gt_annos,
            dt_annos,
            [current_class],
            min_overlaps,
            compute_aos,
            DIForDIS=True)
        return map_3d_r40[0, 1, 0]

    def __getitem__(self, item):
        index = int(self.idx_list[item])
        plane = self._load_ground_plane(index)

        if self.split == 'test':
            # base returns: img, calib.P2, img, info
            inputs, calib, _, info = super().__getitem__(item)
            targets = {
                'ground_plane': plane.astype(np.float32),
                'img_size': np.array(info['img_size'], dtype=np.float32)
            }
            return inputs, calib, targets, info

        inputs, calib, targets, info = super().__getitem__(item)

        # 将平面参数复制到每个 slot，保持与现有 mask 逻辑兼容
        targets['ground_plane'] = np.tile(plane.reshape(1, 4), (self.max_objs, 1)).astype(np.float32)
        targets['img_size'] = np.tile(np.array(info['img_size'], dtype=np.float32).reshape(1, 2),
                                      (self.max_objs, 1)).astype(np.float32)
        return inputs, calib, targets, info

    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)

        test_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

        logger.info('==> Evaluating (official) ...')
        car_moderate = 0
        rope_iou_thresh = 0.5
        rope_ap = {}
        for category in self.writelist:
            if category not in test_id:
                logger.info('==> Skipping official eval for unsupported class: %s', category)
                continue
            results_str, _, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            rope_ap[category] = self._get_ap3d_r40_at_iou(
                gt_annos, dt_annos, test_id[category], rope_iou_thresh)
            if category == 'Car':
                car_moderate = mAP3d_R40
            logger.info(results_str)

        if not os.path.isdir(self.denorm_dir):
            logger.info('==> RopeScore skipped: denorm dir not found: %s', self.denorm_dir)
            return car_moderate

        logger.info('==> Evaluating RopeScore ...')
        rope_ap = {'Car': car_moderate} if 'Car' in self.writelist else {}
        rope_str, _ = get_rope_score_eval_result(
            label_dir=self.label_dir,
            result_dir=results_dir,
            denorm_dir=self.denorm_dir,
            image_ids=self.idx_list,
            class_names=self.writelist,
            ap_3d_r40=rope_ap,
            iou_thresh=rope_iou_thresh)
        logger.info(rope_str)
        return car_moderate

