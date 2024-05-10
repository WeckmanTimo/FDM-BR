import numpy as np
import sys
import time
from scipy.linalg import eigh

d = 1.00
nbase = 10
basis = 'def2-tzvpp'
charge1 = 0
charge2 = 0
spin1 = 0
spin2 = 0
case = 0

systems = ['He', 'Be', 'Ne', 'Mg', 'Ar', 'Ca', 'Kr']

for i in range(len(systems)):
  for j in range(i+1,len(systems)):
    system1 = systems[i]
    system2 = systems[j]

    try:
      BRmat = np.load(f'BRmat_{system1}_{basis}_{nbase}_{charge1}_{spin1}.npz')
      taumat1, Smat1, Pmat1, wmat1, Dmat1 = [BRmat[array] for array in BRmat.files]
      BRmat = np.load(f'BRmat_{system2}_{basis}_{nbase}_{charge2}_{spin2}.npz')
      taumat2, Smat2, Pmat2, wmat2, Dmat2 = [BRmat[array] for array in BRmat.files]

      for dim in [nbase]:
        C6 = 0
        taulist1, tauvecs1 = eigh(taumat1[:dim,:dim], Smat1[:dim,:dim] + Pmat1[:dim,:dim]*d)
        dlist1 = tauvecs1.T.dot(wmat1[:dim] + Dmat1[:dim]*d)

        taulist2, tauvecs2 = eigh(taumat2[:dim,:dim], Smat2[:dim,:dim] + Pmat2[:dim,:dim]*d)
        dlist2 = tauvecs2.T.dot(wmat2[:dim] + Dmat2[:dim]*d)
        dij = 2*np.einsum('i,j->ij', dlist1, dlist2)
        for i_ind in range(dim):
            for j_ind in range(dim):
                C6 += 6 * 0.5 * dij[i_ind,j_ind]**2 / (taulist1[i_ind] + taulist2[j_ind])
        print("%s-%s C6: " % (system1, system2), dim, C6)

    except FileNotFoundError:
      print('File not found for BR_%s_%s_nbase-%d_%d_%d' % (system1, basis, nbase, charge1, spin1))
