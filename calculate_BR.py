import numpy as np
import sys
import time
import os
from xhole import sphbrhole # Import the BR-module

nbase = int(sys.argv[1]) # Dimension of the dispersals
system = sys.argv[2]     # Name of the xyz-file
charge = 0               # Charge of the system
spin = 0                 # Spin of the system

if len(sys.argv) > 3:
  charge = int(sys.argv[3])
  spin = int(sys.argv[4])

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
scheme = quadpy.u3._lebedev.lebedev_031()
nsphere = len(scheme.weights)
azimuthal_polar = scheme.theta_phi.T
azimuthal_polar[:, [1,0]] = azimuthal_polar[:, [0,1]]


# Run a PySCF calculation
from pyscf import gto, tools, scf, dft, cc, mp
from ase.io import read

mol = read(f"{system}.xyz") # Read the xyz-file
pos = mol.get_positions() - mol.get_center_of_mass() # Center the atom/molecule
symb = mol.get_chemical_symbols()

atoms = []
for i in range(len(symb)):
  atoms.append("%s %f %f %f" % (symb[i], pos[i,0], pos[i,1], pos[i,2]))
atomlist = "".join([i + "; " for i in atoms])

basis = 'def2-tzvpp'
mol_pyscf = gto.Mole()
mol_pyscf.build(atom=atomlist, basis=basis, unit="Angstrom", spin=spin, charge=charge)
mf = scf.ROHF(mol_pyscf)
mf.max_cycle = 150
mf.run()

# Make reduced density matrices, use CCSD density
ccsd = cc.CCSD(mf)
ccsd.max_cycle = 250
ccsd.conv_tol = 5e-8
ccsd.conv_tol_normt = 1e-7
ccsd.run()
rdm1 = ccsd.make_rdm1()
#mp2 = mp.MP2(mf).run()
#rdm1 = mp2.make_rdm1()
#mf = dft.UKS(mol_pyscf)
#mf.xc = 'B3LYP'
#mf.run()
#rdm1 = mf.make_rdm1()

# Construct 3D-grid
coords = np.zeros((npoints*nsphere, 3))
for i in range(len(x)):
  for j in range(nsphere):
    coords[i * nsphere + j,0] = x[i] * np.sin(azimuthal_polar[j,1]) * np.cos(azimuthal_polar[j,0])
    coords[i * nsphere + j,1] = x[i] * np.sin(azimuthal_polar[j,1]) * np.sin(azimuthal_polar[j,0])
    coords[i * nsphere + j,2] = x[i] * np.cos(azimuthal_polar[j,1])

rdm1_AOa = np.einsum('ij,ai,bj->ab', rdm1[0], mf.mo_coeff,mf.mo_coeff) # ROHF
rdm1_AOb = np.einsum('ij,ai,bj->ab', rdm1[1], mf.mo_coeff,mf.mo_coeff)
#rdm1_AOa = np.einsum('ij,ai,bj->ab', rdm1[0], mf.mo_coeff[0],mf.mo_coeff[0]) # UHF
#rdm1_AOb = np.einsum('ij,ai,bj->ab', rdm1[1], mf.mo_coeff[1],mf.mo_coeff[1])

# Obtain density and the first and second derivatives of the density on the coords-grid for both spin a and b
aovals = dft.numint.eval_ao(mol_pyscf, coords, deriv=2)
rhoa, rho1a, rho2a, rho3a, nabla2rhoa, taua = dft.numint.eval_rho(mol_pyscf, aovals, rdm1_AOa, xctype='meta-gga')
rhob, rho1b, rho2b, rho3b, nabla2rhob, taub = dft.numint.eval_rho(mol_pyscf, aovals, rdm1_AOb, xctype='meta-gga')
y_a = np.zeros((len(x),))
y_b = np.zeros((len(x),))
Qa = np.zeros((len(x),))
Qb = np.zeros((len(x),))
density_a = np.zeros((len(x),))
density_b = np.zeros((len(x),))

# Construct Q-function used to evaluate the Becke--Roussel parameters and take a spherical average of the density
for i in range(len(x)):
  for j in range(nsphere):
    Qa[i] += 1./6 * (nabla2rhoa[i*nsphere+j] - 2. * (2*taua[i*nsphere+j]-(np.sum(rho1a[i*nsphere+j]**2+rho2a[i*nsphere+j]**2+rho3a[i*nsphere+j]**2)/rhoa[i*nsphere+j]/4.))) * scheme.weights[j]
    Qb[i] += 1./6 * (nabla2rhob[i*nsphere+j] - 2. * (2*taub[i*nsphere+j]-(np.sum(rho1b[i*nsphere+j]**2+rho2b[i*nsphere+j]**2+rho3b[i*nsphere+j]**2)/rhob[i*nsphere+j]/4.))) * scheme.weights[j]
    density_a[i] += np.sqrt(np.sum(rhoa[i*nsphere+j]**2)) * scheme.weights[j]
    density_b[i] += np.sqrt(np.sum(rhob[i*nsphere+j]**2)) * scheme.weights[j]

  if np.isnan(Qa[i]): y_a[i] = 0
  else: y_a[i] += 2./3 * np.pi**(2./3) * density_a[i]**(5./3) / Qa[i]
  if np.isnan(Qb[i]): y_b[i] = 0
  else: y_b[i] += 2./3 * np.pi**(2./3) * density_b[i]**(5./3) / Qb[i]

# Initialize the FDM matrices
Smat = np.zeros((nbase,nbase))
taumat = np.zeros((nbase,nbase))
wmat = np.zeros((nbase,))
Dmat = np.zeros((nbase,))
Pmat = np.zeros((nbase,nbase))
results = np.zeros((7,))

fit_i = np.zeros((2,))
fit_i[0] += 1
# Test of the grid by integrating over the density mediated by the x-hole
# This should sum to the total number of electrons in the system
test = sphbrhole(x, w, azimuthal_polar, scheme.weights, y_a, y_b, density_a, density_b, fit_i, fit_i, results)
print("Test:", test, results[2], results[5], results[6])

# Calculate the FDM matrix elements, load existing ones if available
results *= 0
individual = 0
time0 = time.time()
for i in range(nbase):
  fit_i = np.zeros((nbase+1))
  fit_i[i+1] += 1
  for j in range(i,nbase):
    fit_j = np.zeros((nbase+1))
    fit_j[j+1] += 1
    print(i,j,end='\r')
    time1 = time.time()
    try:
      BRmat = np.load(f'BRmat_{system}_{basis}_{nbase}_{charge}_{spin}.npz')
      taumat_ref, Smat_ref, Pmat_ref, wmat_ref, Dmat_ref = [BRmat[array] for array in BRmat.files]
      results = np.array([wmat_ref[j], taumat_ref[i,j], Smat_ref[i,j], Dmat_ref[j], Dmat_ref[j], Pmat_ref[i,j], Pmat_ref[i,j]])
    except FileNotFoundError:
      results *= 0
      test = sphbrhole(x, w, azimuthal_polar, scheme.weights, y_a, y_b, density_a, density_b, fit_i, fit_j, results)
      individual += 1
    taumat[i,j] = results[1]
    taumat[j,i] = results[1]
    Smat[i,j] = results[2]
    Smat[j,i] = results[2]
    Pmat[i,j] = (results[5] + results[6]) / 2.
    Pmat[j,i] = (results[5] + results[6]) / 2.
    if i==j:
      wmat[i] = results[0]
      Dmat[i] = (results[3] + results[4]) / 2.
    time2 = time.time()

np.savez(f'BRmat_{system}_{basis}_{nbase}_{charge}_{spin}.npz', taumat, Smat, Pmat, wmat, Dmat)

if individual == 0: individual += 1
print("Total time: %.3f   Time per integral: %.3f" % (time2-time0, (time2-time0)/individual))

# Calculate the C6 coefficient using the FDM method with different scale values
from scipy.linalg import eigh
scale = 1.0
for dim in range(1,nbase+1):
  C6 = 0
  taulist, tauvecs = eigh(taumat[:dim,:dim], Smat[:dim,:dim] + Pmat[:dim,:dim]*scale)
  dlist = tauvecs.T.dot(wmat[:dim] + Dmat[:dim]*scale)
  dij = 2*np.einsum('i,j->ij', dlist, dlist)
  for i_ind in range(dim):
    for j_ind in range(dim):
      C6 += 6 * 0.5 * dij[i_ind,j_ind]**2 / (taulist[i_ind] + taulist[j_ind])
  print("BR hole: ", dim, C6, scale)

dim = nbase
for scale in [1.05, 1.10, 1.15, 1.2, 1.25, 1.30]:
  C6 = 0
  taulist, tauvecs = eigh(taumat[:dim,:dim], Smat[:dim,:dim] + Pmat[:dim,:dim]*scale)
  dlist = tauvecs.T.dot(wmat[:dim] + Dmat[:dim]*scale)
  dij = 2*np.einsum('i,j->ij', dlist, dlist)
  for i_ind in range(dim):
    for j_ind in range(dim):
      C6 += 6 * 0.5 * dij[i_ind,j_ind]**2 / (taulist[i_ind] + taulist[j_ind])
  print("BR hole: ", dim, C6, scale)

sys.stdout.flush()
