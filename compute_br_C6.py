import numpy as np
import sys
import time
from scipy.linalg import eigh

nbase = 10
basis = 'def2-tzvpp'
charge = 0
spin = 0
case = 1

systems = ['He', 'Be', 'Ne', 'Mg', 'Ar', 'Ca', 'Kr']

for i in range(len(systems)):
    system = systems[i]
    BRmat = np.load(f'BRmat_{system}_{basis}_{nbase}_{charge}_{spin}.npz')
    taumat, Smat, Pmat, wmat, Dmat = [BRmat[array] for array in BRmat.files]

    for dim in [nbase]:
        C6 = 0
        taulist1, tauvecs1 = eigh(taumat[:dim,:dim], Smat[:dim,:dim] + Pmat[:dim,:dim])
        dlist1 = tauvecs1.T.dot(wmat[:dim] + Dmat[:dim])

        dij = 2*np.einsum('i,j->ij', dlist1, dlist1)
        for i_ind in range(dim):
            for j_ind in range(dim):
                C6 += 6 * 0.5 * dij[i_ind,j_ind]**2 / (taulist1[i_ind] + taulist1[j_ind])
        print("%s C6: " % (system), dim, C6)
