import numpy as np
import sys
import os
import time
import quadpy

case = 0
charge = 0
spin = 0
basis = 'def2-tzvpp'

nbase = int(sys.argv[1]) # Dimension of the dispersals
system = sys.argv[2]     # Name of the atom to calculate
method = sys.argv[3]     # Method to use for the pair-density
charge = 0               # Charge of the system
spin = 0                 # Spin of the system

if len(sys.argv) > 4:
  charge = int(sys.argv[4])
  spin = int(sys.argv[5])

# Set up the radial grid
rrange = 12
npoints = 40
x, w = np.polynomial.laguerre.laggauss(npoints)
w = w / np.exp(-x)
if np.max(x) > rrange:
  c = rrange / np.max(x)
  x *= c
  w *= c

# Set up angular grid
import quadpy
scheme = quadpy.u3._lebedev.lebedev_007()
nsphere = len(scheme.weights)
azimuthal_polar = scheme.theta_phi.T
azimuthal_polar[:, [1,0]] = azimuthal_polar[:, [0,1]]

# Run a PySCF calculation
from pyscf import gto, tools, scf, dft, cc, mp

atomlist = "%s 0 0 0" % system
mol_pyscf = gto.Mole()
mol_pyscf.build(atom=atomlist, basis=basis, unit="Angstrom", spin=spin, charge=charge)
mf = scf.ROHF(mol_pyscf)
mf.max_cycle = 150
mf.run()

if method == 'HF':
  rdm1 = mf.make_rdm1()
  dm = rdm1
  rdm1a = rdm1[0]
  rdm1b = rdm1[1]
  rdm2 = np.einsum('ij,kl->ijkl', rdm1.sum(axis=0), rdm1.sum(axis=0)) - 0.5 * np.einsum('ij,kl->ilkj', rdm1.sum(axis=0), rdm1.sum(axis=0))

elif method == 'CCSD':
  ccsd = cc.CCSD(mf)
  ccsd.max_cycle = 250
  ccsd.conv_tol = 1e-9
  ccsd.conv_tol_normt = 1e-9
  ccsd.run()
  rdm1 = np.einsum('ij,sjk,lk->sil',mf.mo_coeff, np.array(ccsd.make_rdm1()),mf.mo_coeff)
  rdm1a = rdm1[0]
  rdm1b = rdm1[1]
  ccsd_rdm2s = ccsd.make_rdm2()
  ccsd_rdm2 = ccsd_rdm2s[0]+ccsd_rdm2s[1]+ccsd_rdm2s[1].transpose(2, 3, 0, 1)+ccsd_rdm2s[2]
  rdm2 = np.einsum('ai,bj,ijkl,ck,dl->abcd',mf.mo_coeff,mf.mo_coeff,ccsd_rdm2,mf.mo_coeff,mf.mo_coeff, optimize=True)

else:
  if method == 'DFT':
    method = 'LDA'
  mf = dft.RKS(mol_pyscf)
  mf.xc = method
  mf.kernel()
  rdm1 = mf.make_rdm1()
  dm = rdm1
  if spin == 0:
    rdm1a = rdm1/2.
    rdm1b = rdm1/2.
    rdm2 = np.einsum('ij,kl->ijkl', rdm1, rdm1) - 0.5 * np.einsum('ij,kl->ilkj', rdm1, rdm1)
  else:
    rdm1a = rdm1[0]
    rdm1b = rdm1[1]
    rdm2 = np.einsum('ij,kl->ijkl', rdm1.sum(axis=0), rdm1.sum(axis=0)) - 0.5 * np.einsum('ij,kl->ilkj', rdm1.sum(axis=0), rdm1.sum(axis=0))

# Construct 3D-grid
coords = np.zeros((npoints*nsphere, 3))
weights = np.zeros((npoints*nsphere, 1))
for i in range(len(x)):
  for j in range(nsphere):
    coords[i * nsphere + j,0] = x[i] * np.sin(azimuthal_polar[j,1]) * np.cos(azimuthal_polar[j,0])
    coords[i * nsphere + j,1] = x[i] * np.sin(azimuthal_polar[j,1]) * np.sin(azimuthal_polar[j,0])
    coords[i * nsphere + j,2] = x[i] * np.cos(azimuthal_polar[j,1])
    weights[i * nsphere + j] = w[i] * scheme.weights[j] * 4 * np.pi
aovals = dft.numint.eval_ao(mol_pyscf, coords, deriv=0)

# Obtain density on the coords-grid for both spin a and b
rhoa = dft.numint.eval_rho(mol_pyscf, aovals, rdm1a, xctype='LDA')
rhob = dft.numint.eval_rho(mol_pyscf, aovals, rdm1b, xctype='LDA')

rhoa_sph = np.zeros((len(x),))
rhob_sph = np.zeros((len(x),))
aovals_sph = np.zeros((len(x), aovals.shape[1]))

for i in range(len(x)):
  for j in range(nsphere):
    aovals_sph[i,:] += np.sqrt(aovals[i*nsphere+j,:]**2) * scheme.weights[j]
    rhoa_sph[i] += np.sqrt(rhoa[i*nsphere+j]**2) * scheme.weights[j]
    rhob_sph[i] += np.sqrt(rhob[i*nsphere+j]**2) * scheme.weights[j]

from HFpqrs import HFpqrs

time1 = time.time()
N = rdm1.shape[1]
done = 0
loc_bas = 0
individuals = 0

print(system, method, rdm1a.shape[0], npoints, nsphere, npoints*nsphere)
time1 = time.time()

Smat = np.zeros((nbase,nbase))
taumat = np.zeros((nbase,nbase))
wmat = np.zeros((nbase,))
Dmat = np.zeros((nbase,))
Pmat = np.zeros((nbase,nbase))
Kcorr = np.zeros((nbase,nbase))
results = np.zeros((6,))

fit_i = np.zeros((2,))
fit_i[1] += 1

time1 = time.time()
for i in range(nbase):
  fit_i = np.zeros((nbase+1))
  fit_i[i+1] += 1
  for j in range(i, nbase):
    fit_j = np.zeros((nbase+1))
    fit_j[j+1] += 1
    print(i,j,end='\r')
    try:
      for index in range(j,nbase+1):
        if os.path.exists(f'FDMmat_{method}_{system}_{basis}_{nbase}_{charge}_{spin}.npz'):
          FDMmat = np.load(f'FDMmat_{method}_{system}_{basis}_{nbase}_{charge}_{spin}.npz')
          taumat_ref, Smat_ref, Pmat_ref, wmat_ref, Dmat_ref, Kcorr_ref = [FDMmat[array] for array in FDMmat.files]
          break
      results = np.array([wmat_ref[j], taumat_ref[i,j], Smat_ref[i,j], Pmat_ref[i,j], Dmat_ref[j], Kcorr_ref[i,j]])
    except (FileNotFoundError, IndexError, NameError) as e:
      results = np.zeros((6,))
      test = HFpqrs(x, w, azimuthal_polar, scheme.weights, aovals, rdm1a, rdm1b, rdm2, rhoa_sph, rhob_sph, fit_i, fit_j, results)
      individuals += 1
    taumat[i,j] = results[1]
    Kcorr[i,j] = results[5]
    Smat[i,j] = results[2]
    Pmat[i,j] = results[3]

    taumat[j,i] = taumat[i,j]
    Kcorr[j,i] = Kcorr[i,j]
    Smat[j,i] = Smat[i,j]
    Pmat[j,i] = Pmat[i,j]

    if i==j:
      wmat[i] = results[0]
      Dmat[i] = results[4]

if individuals == 0: individuals+=1

time2 = time.time()

np.savez(f'FDMmat_{method}_{system}_{basis}_{nbase}_{charge}_{spin}.npz', taumat, Smat, Pmat, wmat, Dmat, Kcorr)

print("Total time: %.2f seconds with %d individual computations (%.2f sec per integral)." % (time2-time1, individuals, (time2-time1)/individuals))

from scipy.linalg import eigh

for dim in range(1,nbase+1):
    C6 = 0
    taulist, tauvecs = eigh(taumat[:dim,:dim], Smat[:dim,:dim] + Pmat[:dim,:dim])
    dlist = tauvecs.T.dot(wmat[:dim] + Dmat[:dim])
    dij = 2*np.einsum('i,j->ij', dlist, dlist)
    for i_ind in range(dim):
        for j_ind in range(dim):
            C6 += 6 * 0.5 * dij[i_ind,j_ind]**2 / (taulist[i_ind] + taulist[j_ind])
    print("%s C6: " % method, dim, C6)

for dim in range(1,nbase+1):
    C6 = 0
    taulist, tauvecs = eigh(taumat[:dim,:dim] + Kcorr[:dim,:dim], Smat[:dim,:dim] + Pmat[:dim,:dim])
    dlist = tauvecs.T.dot(wmat[:dim] + Dmat[:dim])
    dij = 2*np.einsum('i,j->ij', dlist, dlist)
    for i_ind in range(dim):
        for j_ind in range(dim):
            C6 += 6 * 0.5 * dij[i_ind,j_ind]**2 / (taulist[i_ind] + taulist[j_ind])
    print("%s-K corr C6: " % method, dim, C6)
