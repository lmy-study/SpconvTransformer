def register():
    from pcdet.models import backbones_3d
    from pcdet.models.backbones_2d import map_to_bev
    from .backbone import DualBackbone, FusionBEV
    backbones_3d.__all__['DualBackbone'] = DualBackbone
    map_to_bev.__all__['FusionBEV'] = FusionBEV
