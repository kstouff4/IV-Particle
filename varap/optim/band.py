import torch
import numpy as np
from varap.loss.rescale import rescale_loss
from varap import dtype, fpath, outfile

def optimize(Z, nu_Z, uloss, sig, nb_iter=20, flag='all', rescale=True):
    if flag == 'all':
        x_init = [torch.tensor(Z).type(dtype), torch.tensor(nu_Z).type(dtype).sqrt()]
        dxmax = [sig, x_init[1].mean()]
        print(dxmax)
    else:
        x_init = [torch.tensor(nu_Z).type(dtype).sqrt()]
        print(x_init[0].type())
        dxmax = [x_init[0].mean()]
        print(dxmax[0].type())

    print("x_init ", x_init)
    #  normalize the loss
    beta = uloss(x_init)

    def nuloss(xu):
        return uloss(xu) / beta

    optloss, ufromopt, xopt = rescale_loss(nuloss, x_init, dxmax)
    optimizer = torch.optim.LBFGS(xopt, max_iter=15, line_search_fn='strong_wolfe', history_size=10)

    def closure():
        optimizer.zero_grad()
        L = optloss(xopt)
        print("loss", L.detach().cpu().numpy())
        # print("nu_Z:", ufromopt(xopt)[-1]**2)
        # print("nu_Z max:", (ufromopt(xopt)[-1]**2).max())
        L.backward()
        return L

    for i in range(nb_iter):
        print("it ", i, ": ", end="")
        optimizer.step(closure)
        if flag == 'all':
            xu = ufromopt(xopt)
            nZ = xu[0].detach().cpu().numpy()
            nnu_Z = xu[1].detach().cpu().numpy() ** 2
        else:
            xu = ufromopt(xopt)
            nZ = Z
            nnu_Z = xu[0].detach().cpu().numpy() ** 2
        np.savez_compressed(fpath + outfile, Z=nZ, nu_Z=nnu_Z)

    return nZ, nnu_Z