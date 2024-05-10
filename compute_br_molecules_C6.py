import numpy as np
import sys
import time
from scipy.linalg import eigh

nbase = 8
basis = 'def2-tzvpp'
charge = 0
spin = 0
case = 1

systems = ['H2', 'CH4', 'N2', 'SiH4', 'H2O', 'NH3', 'H2S', 'HF', 'HCl', 'CO', 'H2CO', 'CH3OH' , 'C2H6', 'C2H4', 'C2H2', 'HBr']

for i in range(len(systems)):
    system = systems[i]
    BRmat = np.load(f'BRmat_{system}_{basis}_{nbase}_{charge}_{spin}.npz')
    taumat1, Smat1, Pmat1, wmat1, Dmat1 = [BRmat[array] for array in BRmat.files]

    for dim in [nbase]:
        C6 = 0
        taulist1, tauvecs1 = eigh(taumat1[:dim,:dim], Smat1[:dim,:dim] + Pmat1[:dim,:dim])
        dlist1 = tauvecs1.T.dot(wmat1[:dim] + Dmat1[:dim])

        dij = 2*np.einsum('i,j->ij', dlist1, dlist1)
        for i_ind in range(dim):
            for j_ind in range(dim):
                C6 += 6 * 0.5 * dij[i_ind,j_ind]**2 / (taulist1[i_ind] + taulist1[j_ind])
        print("%s C6: " % (system), dim, C6)
