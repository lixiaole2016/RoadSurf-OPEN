import torch
from torch import nn
import torch.nn.functional as F

from utils.misc import NestedTensor, inverse_sigmoid

from lib.models.monodetr.monodetr import MonoDETR, MLP
from lib.models.monodetr.backbone import build_backbone
from lib.models.monodetr.matcher import build_matcher
from lib.models.monodetr.depthaware_transformer import build_depthaware_transformer
from lib.models.monodetr.depth_predictor import DepthPredictor
from lib.models.roadsurf.ground_field import RoadSurfGroundHead
from lib.models.roadsurf.criterion import RoadSurfCriterion


class RoadSurfMonoDETR(MonoDETR):
    """
    在 MonoDETR 基础上加入路面隐式场预测。
    """

    def __init__(self, road_cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.road_cfg = road_cfg or {}
        self.road_head = RoadSurfGroundHead({'road': road_cfg}, in_channels=self.hidden_dim, hidden_dim=self.hidden_dim)

    def forward(self, images, calibs, targets, img_sizes, dn_args=None):
        features, pos = self.backbone(images)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            if self.training:
                tgt_all_embed = tgt_embed = self.tgt_embed.weight
                refanchor = self.refpoint_embed.weight
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
            else:
                tgt_all_embed = tgt_embed = self.tgt_embed.weight[:self.num_queries]
                refanchor = self.refpoint_embed.weight[:self.num_queries]
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
        elif self.two_stage_dino:
            query_embeds = None
        else:
            if self.training:
                query_embeds = self.query_embed.weight
            else:
                query_embeds = self.query_embed.weight[:self.num_queries]

        pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip = self.depth_predictor(srcs, masks[1], pos[1])

        # 路面隐式场
        B = images.shape[0]
        ground_plane = self._extract_ground_plane(targets, images.device, B)
        road_out = {}
        if ground_plane is not None and ground_plane.shape[0] == images.shape[0]:
            road_out = self.road_head(srcs, calibs, img_sizes, ground_plane)
        else:
            ground_plane = None

        hs, init_reference, inter_references, inter_references_dim, enc_outputs_class, enc_outputs_coord_unact = self.depthaware_transformer(
            srcs, masks, pos, query_embeds, depth_pos_embed, depth_pos_embed_ip)

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)

            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)

            depth_reg = self.depth_embed[lvl](hs[lvl])

            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1),
                outputs_center3d,
                mode='bilinear',
                align_corners=True).squeeze(1)

            depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1) + depth_map) / 3,
                                   depth_reg[:, :, 1: 2]], -1)
            outputs_depths.append(depth_ave)

            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits
        out['img_sizes'] = img_sizes
        out['calibs'] = calibs
        if ground_plane is not None:
            out['ground_plane'] = ground_plane
        out.update(road_out)

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    def _extract_ground_plane(self, targets, device, batch_size):
        if targets is None:
            return None
        if isinstance(targets, (list, tuple)):
            planes = []
            for t in targets:
                plane = t.get('ground_plane', None)
                if plane is None:
                    continue
                if plane.numel() == 0:
                    continue
                if plane.dim() == 3:
                    plane = plane[:, 0, :]  # (b,4)
                elif plane.dim() == 2 and plane.shape[0] > 1:
                    plane = plane[0:1, :]  # (1,4)
                plane = plane.reshape(1, 4)
                planes.append(plane.to(device))
            if len(planes) == 0:
                return None
            planes = torch.stack(planes, dim=0).squeeze(1)
            if planes.shape[0] >= batch_size:
                planes = planes[:batch_size]
            elif planes.shape[0] == 1 and batch_size > 1:
                planes = planes.expand(batch_size, 4)
            else:
                return None
            return planes
        if isinstance(targets, dict):
            plane = targets.get('ground_plane', None)
            if plane is None:
                return None
            if plane.numel() == 0:
                return None
            if plane.dim() == 3:
                plane = plane[:, 0, :]  # (b,4)
            elif plane.dim() == 2:
                if plane.shape[0] == batch_size:
                    pass
                elif plane.shape[0] > 1:
                    plane = plane[:batch_size]
                elif plane.shape[0] == 1 and batch_size > 1:
                    plane = plane.expand(batch_size, 4)
            elif plane.dim() == 1:
                plane = plane.reshape(1, 4).expand(batch_size, 4)
            plane = plane.reshape(-1, 4)
            if plane.shape[0] >= batch_size:
                plane = plane[:batch_size]
            return plane.to(device)
        return None


def build_roadsurf(cfg):
    road_cfg = cfg.get('road', {})
    backbone = build_backbone(cfg)
    depthaware_transformer = build_depthaware_transformer(cfg)
    depth_predictor = DepthPredictor(cfg)

    model = RoadSurfMonoDETR(
        road_cfg,
        backbone,
        depthaware_transformer,
        depth_predictor,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        two_stage=cfg['two_stage'],
        init_box=cfg['init_box'],
        use_dab=cfg['use_dab'],
        two_stage_dino=cfg['two_stage_dino'])

    matcher = build_matcher(cfg)

    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']

    if cfg.get('use_dn'):
        weight_dict['tgt_loss_ce'] = cfg['cls_loss_coef']
        weight_dict['tgt_loss_bbox'] = cfg['bbox_loss_coef']
        weight_dict['tgt_loss_giou'] = cfg['giou_loss_coef']
        weight_dict['tgt_loss_angle'] = cfg['angle_loss_coef']
        weight_dict['tgt_loss_center'] = cfg['3dcenter_loss_coef']

    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # RoadSurf 专用权重（不扩展到 aux）
    weight_dict['loss_road_data'] = road_cfg.get('data_loss_coef', 1.0)
    weight_dict['loss_road_smooth'] = road_cfg.get('smooth_loss_coef', 0.2)
    weight_dict['loss_road_plane'] = road_cfg.get('plane_loss_coef', 0.05)
    weight_dict['loss_road_consistency'] = road_cfg.get('consistency_loss_coef', 1.0)

    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map',
              'road_data', 'road_smooth', 'road_plane', 'road_consistency']

    criterion = RoadSurfCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses,
        road_cfg=road_cfg)

    device = torch.device(cfg['device'])
    criterion.to(device)
    return model, criterion

