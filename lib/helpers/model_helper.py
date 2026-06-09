from lib.models.monodetr import build_monodetr


def build_model(cfg):
    """
    根据配置选择基础 MonoDETR 或 RoadSurf 变体。
    """
    if cfg.get('use_roadsurf', False):
        from lib.models.roadsurf import build_roadsurf
        return build_roadsurf(cfg)
    return build_monodetr(cfg)
