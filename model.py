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
        const_beta: float = -1,
    ):
        super().__init__(nnodes, dim, scale, global_ll)
        self.with_base_intensity = with_base_intensity
        self.const_beta = const_beta

    def initialize_weights(self, alpha_scale=-15):
        if self.with_base_intensity:
            self.mus_.weight.data.fill_(-5 * self.scale)  # NJ
        else:
            self.mus_.weight.data.fill_(-1e10)

        self.beta_.data.fill_(5)  # NJ
        if self.const_beta > 0:
            self.beta_.data.fill_(self.const_beta)
        self.self_A.weight.data.fill_(0)  # NJ
        # Randomly initialize embeddings, except for the pad embedding
        self.U.weight.data[:-1].copy_(
            # th.rand(self.nnodes, self.dim, dtype=th.double)
            # * (-5 * self.scale)  # NYC
            th.rand(self.nnodes, self.dim, dtype=th.double)
            * (alpha_scale * self.scale)  # NJ
        )
        self.V.weight.data[:-1].copy_(
            # th.rand(self.nnodes, self.dim, dtype=th.double)
            # * (-5 * self.scale)  # NYC
            th.rand(self.nnodes, self.dim, dtype=th.double)
            * (alpha_scale * self.scale)  # NJ
        )
        # Set these to a large negative value, such that fpos(pad_emb) == 0
        self.U.weight.data[-1].fill_(-1e10)
        self.V.weight.data[-1].fill_(-1e10)
        self.self_A.weight.data[-1].fill_(-1e10)
        self.mus_.weight.data[-1].fill_(-1e10)

    def mus(self, x):
        m = self.fpos(self.mus_(x))
        if not self.with_base_intensity:
            m = m * 0
        return m

    def beta(self):
        b = self.fpos(self.beta_)
        if self.const_beta > 0:
            b = b * 0 + self.const_beta
        return b

    def alpha(self, xs: th.LongTensor, ys: th.LongTensor):
        U = self.fpos(self.U(xs))
        V = self.fpos(self.V(ys))
        res = th.bmm(U.view(-1, 1, U.size(-1)), V.view(-1, V.size(-1), 1))
        res = res.view(U.shape[:-1]) / self.dim
        mask = xs == ys
        res[mask] = self.fpos(self.self_A(xs).squeeze(-1)[mask])
        return res
