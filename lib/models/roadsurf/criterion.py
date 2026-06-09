import torch
import torch.nn.functional as F

from lib.models.monodetr.monodetr import SetCriterion


class RoadSurfCriterion(SetCriterion):
    """
    在原有 SetCriterion 基础上，加入路面隐式场的监督与一致性约束。
    """

    def __init__(self, *args, road_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.road_cfg = road_cfg or {}
        self.xbound = self.road_cfg.get('xbound', [-30.0, 30.0, 0.5])
        self.zbound = self.road_cfg.get('zbound', [0.0, 80.0, 0.5])
        self.solve_axis = self.road_cfg.get('plane_solve_axis', 'beta')

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if loss == 'road_data':
            return self.loss_road_data(outputs, targets)
        if loss == 'road_smooth':
            return self.loss_road_smooth(outputs)
        if loss == 'road_plane':
            return self.loss_road_plane(outputs)
        if loss == 'road_consistency':
            return self.loss_road_consistency(outputs, targets, indices)
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)

    def loss_road_data(self, outputs, targets):
        residual_map = outputs.get('road_residual')
        ground_plane = outputs.get('ground_plane')
        img_sizes = outputs.get('img_sizes')
        if residual_map is None or ground_plane is None or img_sizes is None:
            device = next(iter(outputs.values())).device
            return {'loss_road_data': torch.tensor(0.0, device=device)}

        bs = residual_map.shape[0]
        losses = []
        for b in range(bs):
            tgt = targets[b]
            plane = ground_plane[b]
            boxes = tgt.get('boxes_3d')
            depth = tgt.get('depth')
            calibs = tgt.get('calibs')
            if boxes is None or depth is None or calibs is None or boxes.numel() == 0:
                continue

            u = boxes[:, 0] * img_sizes[b, 0]
            v = boxes[:, 1] * img_sizes[b, 1]
            z = depth[:, 0]
            x, y, z = self.img_to_rect(u, v, z, calibs, bidx=None)

            plane_h = self.plane_height(plane, x, z)
            gt_residual = y - plane_h

            inside = (x >= self.xbound[0]) & (x <= self.xbound[1]) & (z >= self.zbound[0]) & (z <= self.zbound[1])
            if inside.sum() == 0:
                continue

            pred_res = self.sample_from_map(residual_map[b:b + 1], x[inside], z[inside])
            losses.append(F.l1_loss(pred_res.squeeze(0), gt_residual[inside], reduction='mean'))

        if len(losses) == 0:
            return {'loss_road_data': residual_map.sum() * 0}
        return {'loss_road_data': torch.stack(losses).mean()}

    def loss_road_smooth(self, outputs):
        residual_map = outputs.get('road_residual')
        if residual_map is None:
            return {'loss_road_smooth': torch.tensor(0.0, device=next(iter(outputs.values())).device)}
        dx = residual_map[:, :, :, 1:] - residual_map[:, :, :, :-1]
        dz = residual_map[:, :, 1:, :] - residual_map[:, :, :-1, :]
        loss = dx.abs().mean() + dz.abs().mean()
        return {'loss_road_smooth': loss}

    def loss_road_plane(self, outputs):
        residual_map = outputs.get('road_residual')
        if residual_map is None:
            return {'loss_road_plane': torch.tensor(0.0, device=next(iter(outputs.values())).device)}
        return {'loss_road_plane': residual_map.abs().mean()}

    def loss_road_consistency(self, outputs, targets, indices):
        residual_map = outputs.get('road_residual')
        road_height = outputs.get('road_height')
        ground_plane = outputs.get('ground_plane')
        img_sizes = outputs.get('img_sizes')
        calibs = outputs.get('calibs')
        if residual_map is None or road_height is None or ground_plane is None or img_sizes is None or calibs is None:
            device = next(iter(outputs.values())).device
            return {'loss_road_consistency': torch.tensor(0.0, device=device)}

        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {'loss_road_consistency': residual_map.sum() * 0}

        pred_boxes = outputs['pred_boxes'][idx]
        pred_depth = outputs['pred_depth'][idx][:, 0]
        batch_ids = idx[0]

        losses = []
        for b in range(residual_map.shape[0]):
            mask_b = batch_ids == b
            if mask_b.sum() == 0:
                continue
            u = pred_boxes[mask_b, 0] * img_sizes[b, 0]
            v = pred_boxes[mask_b, 1] * img_sizes[b, 1]
            depth_b = pred_depth[mask_b]
            x, y, z = self.img_to_rect(u, v, depth_b, calibs, bidx=b)
            plane_h = self.plane_height(ground_plane[b], x, z)
            pred_surface = self.sample_from_map(road_height[b:b + 1], x, z).squeeze(0)
            losses.append(F.l1_loss(y, pred_surface, reduction='mean') + F.l1_loss(plane_h, pred_surface, reduction='mean'))

        if len(losses) == 0:
            return {'loss_road_consistency': residual_map.sum() * 0}
        return {'loss_road_consistency': torch.stack(losses).mean()}

    def img_to_rect(self, u, v, depth, calib, bidx=None):
        """
        基于 P2 内参将 (u,v,depth) 转为相机坐标 (x,y,z)
        """
        if bidx is None:
            fu = calib[:, 0, 0]
            fv = calib[:, 1, 1]
            cu = calib[:, 0, 2]
            cv = calib[:, 1, 2]
            tx = calib[:, 0, 3] / (-fu)
            ty = calib[:, 1, 3] / (-fv)
        else:
            fu = calib[bidx, 0, 0]
            fv = calib[bidx, 1, 1]
            cu = calib[bidx, 0, 2]
            cv = calib[bidx, 1, 2]
            tx = calib[bidx, 0, 3] / (-fu)
            ty = calib[bidx, 1, 3] / (-fv)
        x = ((u - cu) * depth) / fu + tx
        y = ((v - cv) * depth) / fv + ty
        return x, y, depth

    def plane_height(self, plane, x, z):
        alpha = plane[..., 0]
        beta = plane[..., 1]
        gamma = plane[..., 2]
        d = plane[..., 3]
        if self.solve_axis == 'gamma':
            return -(alpha * x + beta * z + d) / (gamma + 1e-6)
        return -(alpha * x + gamma * z + d) / (beta + 1e-6)

    def sample_from_map(self, feature_map, x, z):
        """
        从 BEV 特征图中双线性插值采样 (x,z) 对应位置。
        feature_map: (B, 1, Hz, Wx) 或 (Hz, Wx)
        """
        if feature_map.dim() == 3:
            feature_map = feature_map.unsqueeze(0)
        B, C, H, W = feature_map.shape
        x_norm = (x - self.xbound[0]) / (self.xbound[1] - self.xbound[0]) * 2 - 1
        z_norm = (z - self.zbound[0]) / (self.zbound[1] - self.zbound[0]) * 2 - 1
        grid = torch.stack([x_norm, z_norm], dim=-1).view(B, -1, 1, 2)
        sampled = F.grid_sample(feature_map, grid, align_corners=True, padding_mode='zeros').view(B, -1)
        return sampled

