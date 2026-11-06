"""Coordinate lookup attention: O(N K log N), without dense all-pairs matrices."""
import itertools
import torch
from torch import nn

def keys(coords, shape):
    z, y, x = shape
    c = coords.long()
    return ((c[:, 0] * z + c[:, 1]) * y + c[:, 2]) * x + c[:, 3]

class VoxelAttention(nn.Module):
    def __init__(self, channels, heads=4, chunk_size=2048):
        super().__init__()
        self.heads, self.chunk_size = heads, chunk_size
        self.qkv = nn.Linear(channels, channels * 3)
        self.position = nn.Sequential(nn.Linear(3, channels), nn.ReLU(), nn.Linear(channels, heads))
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))
        offsets = {(0,0,0)}
        for stride in (1, 2, 4):
            offsets.update(tuple(stride*v for v in o) for o in itertools.product((-1,0,1), repeat=3))
        self.register_buffer('offsets', torch.tensor(sorted(offsets), dtype=torch.long))

    def forward(self, features, coords, shape, residual=True):
        if len(features) == 0:
            return features
        n, c = features.shape
        h, d = self.heads, c // self.heads
        q,k,v = self.qkv(features).reshape(n,3,h,d).unbind(1)
        sorted_keys, order = keys(coords, shape).sort()
        outputs = []
        for start in range(0,n,self.chunk_size):
            end = min(start+self.chunk_size,n)
            query = coords[start:end].long()
            xyz = query[:,None,1:] + self.offsets[None]
            valid = ((xyz >= 0) & (xyz < xyz.new_tensor(shape))).all(-1)
            candidate = torch.cat((query[:,None,:1].expand(-1,len(self.offsets),-1), xyz),-1)
            target = keys(candidate.reshape(-1,4),shape).reshape(len(query),-1)
            loc = torch.searchsorted(sorted_keys,target).clamp(max=n-1)
            valid &= sorted_keys[loc] == target
            idx = order[loc]
            score = (q[start:end,None]*k[idx]).sum(-1)/(d**0.5)
            score = score + self.position(self.offsets.to(features.dtype))[None]
            score = score.masked_fill(~valid[:,:,None],float('-inf'))
            outputs.append((score.softmax(1)[...,None]*v[idx]).sum(1).reshape(-1,c))
        out = self.proj(torch.cat(outputs))
        if residual:
            out = out + features
        return out + self.ffn(self.norm(out))
