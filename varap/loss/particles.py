import torch
from pykeops.torch import Vi, Vj
from varap.loss.rescale import rescale_loss
from varap import dtype


class ParticleLoss_full():
    def __init__(self, sig, X, nu_X, bw=-1):

        self.dtype = dtype

        self.sig = sig
        self.X = X
        self.nu_X = nu_X

        self.loss = self.make_loss() if (bw < 0) else self.slice_it(bw)

    def make_loss(self, X, nu_X):
        tx = torch.tensor(X).type(self.dtype).contiguous()
        LX_i, LX_j = Vi(tx), Vj(tx)

        tnu_X = torch.tensor(nu_X).type(self.dtype).contiguous()
        Lnu_X_i, Lnu_X_j = Vi(tnu_X), Vj(tnu_X)

        D_ij = ((LX_i - LX_j) ** 2 / self.sig ** 2).sum(dim=2)
        K_ij = (- D_ij).exp()
        P_ij = (Lnu_X_i * Lnu_X_j).sum(dim=2)
        c = (K_ij * P_ij).sum(dim=1).sum()
        print('c=', c)

        def loss(tZal_Z):
            LZ_i, LZ_j = Vi(tZal_Z[0]), Vj(tZal_Z[0])

            Lnu_Z_i = Vi(tZal_Z[1] ** 2)
            Lnu_Z_j = Vj(tZal_Z[1] ** 2)

            DZZ_ij = ((LZ_i - LZ_j) ** 2 / self.sig ** 2).sum(dim=2)
            KZZ_ij = (- DZZ_ij).exp()
            PZZ_ij = (Lnu_Z_i * Lnu_Z_j).sum(dim=2)

            DZX_ij = ((LZ_i - LX_j) ** 2 / self.sig ** 2).sum(dim=2)
            KZX_ij = (- DZX_ij).exp()
            PZX_ij = (Lnu_Z_i * Lnu_X_j).sum(dim=2)

            E = (KZZ_ij * PZZ_ij).sum(dim=1) - 2 * (KZX_ij * PZX_ij).sum(dim=1)
            L = E.sum() + c
            return L

        return loss

    def eval(self, *args):
        return self.loss(*args)

    def slice_it(self, bw):

        nb_bands = int(self.nu_X.shape[1] / bw) + 1
        bands = [(i * bw, min((i + 1) * bw, self.nu_X.shape[1])) for i in range(nb_bands)]

        ltmploss = [self.make_loss(self.X,
                                   self.nu_X[:, bands[i][0]:bands[i][1]]) for i in range(nb_bands)]

        def uloss(xu):
            return sum([ltmploss[i]([xu[0], xu[1][:, bands[i][0]:bands[i][1]]]) for i in range(nb_bands)])

        return uloss



class ParticleLoss_restricted():

    def __init__(self, sig, X, nu_X, Z, bw=-1):
        self.sig = sig
        self.X = X
        self.Z = Z
        self.nu_X = nu_X

        self.loss = self.make_loss() if (bw < 0) else self.slice_it(bw)

    def make_loss(self, X, nu_X, Z):
        tx = torch.tensor(X).type(dtype).contiguous()
        LX_i = Vi(tx)
        LX_j = Vj(tx)
        Lnu_X_i = Vi(torch.tensor(nu_X).type(dtype).contiguous())
        Lnu_X_j = Vj(torch.tensor(nu_X).type(dtype).contiguous())

        D_ij = ((LX_i - LX_j) ** 2 / self.sig ** 2).sum(dim=2)
        K_ij = (- D_ij).exp()
        P_ij = (Lnu_X_i * Lnu_X_j).sum(dim=2)
        c = (K_ij * P_ij).sum(dim=1).sum()

        tz = torch.tensor(Z).type(dtype).contiguous()
        LZ_i, LZ_j = Vi(tz), Vj(tz)
        DZZ_ij = ((LZ_i - LZ_j) ** 2 / self.sig ** 2).sum(dim=2)
        KZZ_ij = (- DZZ_ij).exp()

        print('c=', c)


        def loss(tal_Z):
            Lnu_Z_i, Lnu_Z_j = Vi(tal_Z[0] ** 2), Vj(tal_Z[0] ** 2)

            PZZ_ij = (Lnu_Z_i * Lnu_Z_j).sum(dim=2)

            DZX_ij = ((LZ_i - LX_j) ** 2 / self.sig ** 2).sum(dim=2)
            KZX_ij = (- DZX_ij).exp()
            PZX_ij = (Lnu_Z_i * Lnu_X_j).sum(dim=2)

            E = (KZZ_ij * PZZ_ij).sum(dim=1) - 2 * (KZX_ij * PZX_ij).sum(dim=1)
            L = E.sum() + c
            return L

        return loss

    def eval(self, *args):
        return self.loss(*args)

    def slice_it(self, bw):
        nb_bands = int(self.nu_X.shape[1] / bw) + 1
        bands = [(i * bw, min((i + 1) * bw, self.nu_X.shape[1])) for i in range(nb_bands)]

        ltmploss = [self.make_loss(self.X,
                               self.nu_X[:, bands[i][0]:bands[i][1]],
                               self.Z) for i in range(nb_bands)]

        def uloss(xu):
            return sum([ltmploss[i]([xu[0][:, bands[i][0]:bands[i][1]]]) for i in range(nb_bands)])

        return uloss



