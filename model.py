import torch as th
import torch.nn.functional as F
import tl_kernels
from torch.autograd import Function
from timelord.ll import SparseEmbeddingSoftplus


class CovidModel(SparseEmbeddingSoftplus):
    def __init__(
        self,
        nnodes: int,
        dim: int,
        scale: float = 1,
        global_ll: bool = True,
        with_base_intensity: bool = True,
    ):
        super().__init__(nnodes, dim, scale, global_ll)
        self.with_base_intensity = with_base_intensity

    def initialize_weights(self):
        self.mus_.weight.data.fill_(-5 * self.scale)
        self.beta_.data.fill_(0)
        self.self_A.weight.data.fill_(-5 * self.scale)
        # Randomly initialize embeddings, except for the pad embedding
        self.U.weight.data[:-1].copy_(
            th.rand(self.nnodes, self.dim, dtype=th.double) * (-10 * self.scale)
        )
        self.V.weight.data[:-1].copy_(
            th.rand(self.nnodes, self.dim, dtype=th.double) * (-10 * self.scale)
        )
        # Set these to a large negative value, such that fpos(pad_emb) == 0
        self.U.weight.data[-1].fill_(-1e10)
        self.V.weight.data[-1].fill_(-1e10)
        self.self_A.weight.data[-1].fill_(-1e10)
        self.mus_.weight.data[-1].fill_(-1e10)

    def mus(self, x):
        m = self.fpos(self.mus_(x))
        if not self.with_base_intensity:
            m *= 0
        return m

    def beta(self):
        return self.fpos(self.beta_)

    def alpha(self, xs: th.LongTensor, ys: th.LongTensor):
        U = self.fpos(self.U(xs))
        V = self.fpos(self.V(ys))
        res = th.bmm(U.view(-1, 1, U.size(-1)), V.view(-1, V.size(-1), 1))
        res = res.view(U.shape[:-1]) / self.dim
        mask = xs == ys
        res[mask] = self.fpos(self.self_A(xs).squeeze(-1)[mask])
        return res
