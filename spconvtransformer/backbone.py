import torch
from torch import nn
import spconv.pytorch as spconv
from pcdet.models.backbones_3d.spconv_backbone import VoxelBackBone8x
from .attention import VoxelAttention

class DualBackbone(nn.Module):
    """Independent attention branch; established SECOND sparse CNN branch."""
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.cnn = VoxelBackBone8x(model_cfg,input_channels,grid_size)
        self.shape = self.cnn.sparse_shape
        self.embedding = nn.Linear(input_channels,16)
        self.down = nn.ModuleList()
        self.attention = nn.ModuleList()
        for i,(a,b) in enumerate(((16,32),(32,64),(64,64))):
            # Sparse convolution generates new query support and initial query features.
            self.down.append(spconv.SparseConv3d(a,b,3,stride=2,
                padding=1 if i<2 else (0,1,1),bias=False,indice_key=f'attn_down{i}'))
            self.attention.append(nn.ModuleList([VoxelAttention(b) for _ in range(3)]))
        self.num_point_features = 128

    def forward(self,batch_dict):
        x = spconv.SparseConvTensor(self.embedding(batch_dict['voxel_features']),
            batch_dict['voxel_coords'].int(),self.shape,batch_dict['batch_size'])
        for down,blocks in zip(self.down,self.attention):
            x = down(x)
            for i,block in enumerate(blocks):
                x = x.replace_feature(block(x.features,x.indices,x.spatial_shape,residual=i>0))
        batch_dict = self.cnn(batch_dict)
        batch_dict['transformer_tensor'] = x
        return batch_dict

class FusionBEV(nn.Module):
    def __init__(self,model_cfg,**kwargs):
        super().__init__()
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES
    def forward(self,batch_dict):
        a = batch_dict['encoded_spconv_tensor'].dense().flatten(1,2)
        b = batch_dict['transformer_tensor'].dense().flatten(1,2)
        if a.shape[-2:] != b.shape[-2:]:
            raise ValueError('Branch BEV grids do not align')
        fused = torch.cat((a,b),dim=1)
        if fused.shape[1] != self.num_bev_features:
            raise ValueError('Configuration requires 576 BEV channels; verify voxel grid')
        batch_dict['spatial_features'] = fused
        batch_dict['spatial_features_stride'] = 8
        return batch_dict
