"""GPU forward/backward backbone smoke; no KITTI data or training claims."""
import numpy as np
import torch
from easydict import EasyDict
from spconvtransformer.backbone import DualBackbone, FusionBEV
if not torch.cuda.is_available():
    raise SystemExit('CUDA required for spconv integration smoke')
m=DualBackbone(EasyDict(),4,np.array([32,32,40])).cuda().eval()
coords=torch.cartesian_prod(torch.arange(1),torch.arange(40),torch.arange(0,32,4),torch.arange(0,32,4)).int().cuda()
f=torch.randn(len(coords),4,device='cuda',requires_grad=True)
out=m(dict(voxel_features=f,voxel_coords=coords,batch_size=1))
y=FusionBEV(EasyDict(NUM_BEV_FEATURES=576))(out)['spatial_features']
assert y.shape == (1,576,4,4),y.shape
y.square().mean().backward()
assert torch.isfinite(f.grad).all()
print('CUDA dual backbone forward/backward passed')
