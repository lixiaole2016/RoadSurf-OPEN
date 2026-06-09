import torch
import torch.nn as nn
import torch.nn.functional as F


class RoadSurfGroundHead(nn.Module):
    """
    基于 denorm 平面 + 残差的 2.5D 路面高度场。
    1) 使用平面估计将特征 IPM 到 BEV；
    2) 小型 CNN 预测残差 Delta z；
    3) 输出残差、高度图与有效 mask。
    """

    def __init__(self, cfg, in_channels, hidden_dim):
        super().__init__()
        road_cfg = cfg.get('road', {})
        self.xbound = road_cfg.get('xbound', [-30.0, 30.0, 0.5])
        self.zbound = road_cfg.get('zbound', [0.0, 80.0, 0.5])
        self.feat_level = road_cfg.get('feat_level', 1)
        self.solve_axis = road_cfg.get('plane_solve_axis', 'beta')  # 'beta' or 'gamma'

        self.x_steps = int((self.xbound[1] - self.xbound[0]) / self.xbound[2])
        self.z_steps = int((self.zbound[1] - self.zbound[0]) / self.zbound[2])

        conv_in = in_channels + 4  # 拼接平面参数
        channels = road_cfg.get('road_channels', hidden_dim)
        num_convs = road_cfg.get('road_num_convs', 3)
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(conv_in if i == 0 else channels, channels, 3, padding=1))
            layers.append(nn.GroupNorm(32, channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.conv = nn.Sequential(*layers)

    def forward(self, feats, calibs, img_sizes, planes):
        """
        Args:
            feats: list of FPN特征 (B, C, Hf, Wf)
            calibs: (B, 3, 4) 投影矩阵
            img_sizes: (B, 2) 原图尺寸 (w, h)
            planes: (B, 4) denorm 平面参数
        Returns:
            dict 包含 road_residual, road_height, road_valid_mask
        """
        feat = feats[self.feat_level]
        B, C, Hf, Wf = feat.shape
        device = feat.device

        x_lin = torch.linspace(self.xbound[0] + self.xbound[2] / 2,
                               self.xbound[1] - self.xbound[2] / 2,
                               self.x_steps, device=device)
        z_lin = torch.linspace(self.zbound[0] + self.zbound[2] / 2,
                               self.zbound[1] - self.zbound[2] / 2,
                               self.z_steps, device=device)
        grid_z, grid_x = torch.meshgrid(z_lin, x_lin, indexing='ij')
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_z = grid_z.unsqueeze(0).expand(B, -1, -1)

        planes = planes.to(device)
        alpha = planes[:, 0].view(B, 1, 1)
        beta = planes[:, 1].view(B, 1, 1)
        gamma = planes[:, 2].view(B, 1, 1)
        d = planes[:, 3].view(B, 1, 1)

        if self.solve_axis == 'gamma':
            # z = -(ax + by + d) / c
            height = -(alpha * grid_x + beta * grid_z + d) / (gamma + 1e-6)
        else:
            # y = -(ax + cz + d) / b
            height = -(alpha * grid_x + gamma * grid_z + d) / (beta + 1e-6)

        ones = torch.ones_like(grid_x)
        points = torch.stack([grid_x, height, grid_z, ones], dim=-1)  # (B, Hz, Wx, 4)
        proj = torch.einsum('bij,bhwj->bhwi', calibs, points)
        u = proj[..., 0] / proj[..., 2].clamp(min=1e-6)
        v = proj[..., 1] / proj[..., 2].clamp(min=1e-6)

        img_w = img_sizes[:, 0].view(B, 1, 1).clamp(min=1.0)
        img_h = img_sizes[:, 1].view(B, 1, 1).clamp(min=1.0)
        stride_w = img_w / float(Wf)
        stride_h = img_h / float(Hf)
        u_feat = u / stride_w
        v_feat = v / stride_h

        grid_u = (u_feat / (Wf - 1)).clamp(min=-1e4, max=1e4) * 2 - 1
        grid_v = (v_feat / (Hf - 1)).clamp(min=-1e4, max=1e4) * 2 - 1
        sample_grid = torch.stack([grid_u, grid_v], dim=-1)  # (B, Hz, Wx, 2)
        bev_feat = F.grid_sample(feat, sample_grid, align_corners=True)

        # 平面参数拼接
        plane_feat = torch.stack([alpha, beta, gamma, d], dim=1).expand(-1, -1, self.z_steps, self.x_steps)
        inp = torch.cat([bev_feat, plane_feat], dim=1)
        residual = self.conv(inp)
        height_map = height.unsqueeze(1) + residual

        valid_mask = (grid_u.abs() <= 1) & (grid_v.abs() <= 1)
        return {
            'road_residual': residual,
            'road_height': height_map,
            'road_valid_mask': valid_mask
        }

