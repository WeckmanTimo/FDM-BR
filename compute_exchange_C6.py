import numpy as np
import sys
import time
from scipy.linalg import eigh

nbase = 6
basis = 'def2-tzvpp'
method = 'HF'
#method = 'CCSD'
#method = 'LDA'
#method = 'PBE'
charge = 0
spin = 0
case = 1

systems = ['He', 'Be', 'Ne', 'Mg', 'Ar', 'Ca', 'Kr']

for i in range(len(systems)):
    system = systems[i]
    BRmat = np.load(f'FDMmat_{method}_{system}_{basis}_{nbase}_{charge}_{spin}.npz')
    taumat, Smat, Pmat, wmat, Dmat, Kcorr = [BRmat[array] for array in BRmat.files]

    # FDM procedure without the exchange correction
    for dim in range(1,nbase+1):
        C6 = 0
        taulist1, tauvecs1 = eigh(taumat[:dim,:dim], Smat[:dim,:dim] + Pmat[:dim,:dim])
        dlist1 = tauvecs1.T.dot(wmat[:dim] + Dmat[:dim])

        dij = 2*np.einsum('i,j->ij', dlist1, dlist1)
        for i_ind in range(dim):
            for j_ind in range(dim):
                C6 += 6 * 0.5 * dij[i_ind,j_ind]**2 / (taulist1[i_ind] + taulist1[j_ind])
        print("%s %s C6: " % (method, system), dim, C6)

    # FDM procedure with the exchange correction
    for dim in range(1,nbase+1):
        C6 = 0
        taulist1, tauvecs1 = eigh(taumat[:dim,:dim] + Kcorr[:dim,:dim], Smat[:dim,:dim] + Pmat[:dim,:dim])
        dlist1 = tauvecs1.T.dot(wmat[:dim] + Dmat[:dim])

        dij = 2*np.einsum('i,j->ij', dlist1, dlist1)
        for i_ind in range(dim):
            for j_ind in range(dim):
                C6 += 6 * 0.5 * dij[i_ind,j_ind]**2 / (taulist1[i_ind] + taulist1[j_ind])
        if method == 'HF' or method == 'LDA' or method == 'PBE':
            print("%s-K %s C6: " % (method, system), dim, C6)
