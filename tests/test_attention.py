import unittest
import torch
from spconvtransformer.attention import VoxelAttention
class AttentionTests(unittest.TestCase):
    def test_batch_isolation_and_gradient(self):
        torch.manual_seed(7)
        m=VoxelAttention(16,chunk_size=2)
        coords=torch.tensor([[0,0,0,0],[0,0,0,1],[1,0,0,0]])
        x=torch.randn(3,16,requires_grad=True)
        y=m(x,coords,(3,3,3))
        changed=x.detach().clone(); changed[2]+=100
        self.assertTrue(torch.allclose(y[:2],m(changed,coords,(3,3,3))[:2]))
        self.assertTrue(torch.isfinite(y).all())
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
    def test_empty(self):
        self.assertEqual(VoxelAttention(16)(torch.empty(0,16),torch.empty(0,4,dtype=torch.long),(3,3,3)).shape,(0,16))
if __name__=='__main__': unittest.main()

class EquivarianceTests(unittest.TestCase):
    def test_permutation_and_chunk_equivalence(self):
        torch.manual_seed(13)
        m = VoxelAttention(16, chunk_size=1)
        x = torch.randn(4,16)
        c = torch.tensor([[0,0,0,0],[0,0,0,1],[0,1,0,0],[0,2,2,2]])
        a = m(x,c,(3,3,3))
        m.chunk_size=100
        self.assertTrue(torch.allclose(a,m(x,c,(3,3,3)),atol=1e-6))
        p=torch.tensor([2,0,3,1])
        self.assertTrue(torch.allclose(a[p],m(x[p],c[p],(3,3,3)),atol=1e-6))
