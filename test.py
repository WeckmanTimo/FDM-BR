import numpy as np
import sys
import time
from pyscf import gto, tools, scf, dft, cc
import quadpy
#from brhole import brholegrid
from brhole import brholesto6g as brholegrid

nbase = int(sys.argv[1])
system = sys.argv[2]
charge = 0
spin = 0

if len(sys.argv) > 3:
  charge = int(sys.argv[3])
  spin = int(sys.argv[4])

# Compute the needed multipoles
bbasis = []
for s in range(nbase):
  for t in range(nbase-s):
    for u in range(nbase-s-t):
      if (s+t+u)%2 == 1:
        bbasis.append([s, t, u])
bbasis = np.array(bbasis)
nb = len(bbasis)

# For spherical systems (or e.g. H2 in the correct alignment) use three mirror plane symmetries to constrain b.
rrange = 8
npoints = 16

# Grid for the radial part
x, w = np.polynomial.laguerre.laggauss(npoints)
w = w / np.exp(-x)
if np.max(x) > rrange:
  c = rrange / np.max(x)
  x *= c
  w *= c
# Lebedev-Laikov grid for the angular part
scheme = quadpy.sphere.lebedev_015()
nsphere = len(scheme.weights)

# Run a HF calculation
atomlist = "%s 0 0 0" % system
basis = 'def2-tzvpp'
mol_pyscf = gto.Mole()
mol_pyscf.build(atom=atomlist, basis=basis, unit="Angstrom", spin=spin, charge=charge)
mf = scf.ROHF(mol_pyscf)
mf.max_cycle = 150
mf.run()

# Make reduced density matrices
ccsd = cc.CCSD(mf).run()
rdm1 = ccsd.make_rdm1()

#Construct a 3D grid
coords = np.zeros((npoints*nsphere, 3))
for i in range(len(x)):
  for j in range(nsphere):
    coords[i * nsphere + j,0] = x[i] * np.sin(scheme.azimuthal_polar[j,1]) * np.cos(scheme.azimuthal_polar[j,0])
    coords[i * nsphere + j,1] = x[i] * np.sin(scheme.azimuthal_polar[j,1]) * np.sin(scheme.azimuthal_polar[j,0])
    coords[i * nsphere + j,2] = x[i] * np.cos(scheme.azimuthal_polar[j,1])

rdm1_AOa = np.einsum('ij,ai,bj->ab', rdm1[0], mf.mo_coeff,mf.mo_coeff)
rdm1_AOb = np.einsum('ij,ai,bj->ab', rdm1[1], mf.mo_coeff,mf.mo_coeff)
# Obtain the atomic orbitals and derivatives on the 3D grid
aovals = dft.numint.eval_ao(mol_pyscf, coords, deriv=2)
rhoa, rho1a, rho2a, rho3a, nabla2rhoa, taua = dft.numint.eval_rho(mol_pyscf, aovals, rdm1_AOa, xctype='meta-gga')
rhob, rho1b, rho2b, rho3b, nabla2rhob, taub = dft.numint.eval_rho(mol_pyscf, aovals, rdm1_AOb, xctype='meta-gga')
rho = rhoa+rhob
# Construct y_a/y_b functions for the Becke-Roussel parametrization
y_a = np.zeros((len(x),))
y_b = np.zeros((len(x),))
Qa = np.zeros((len(x),))
Qb = np.zeros((len(x),))
density_a = np.zeros((len(x),))
density_b = np.zeros((len(x),))
for i in range(len(x)):
  for j in range(nsphere):
    Qa[i] += 1./6 * (nabla2rhoa[i*nsphere+j] - 2. * (2*taua[i*nsphere+j]-(np.sum(rho1a[i*nsphere+j]**2+rho2a[i*nsphere+j]**2+rho3a[i*nsphere+j]**2)/rhoa[i*nsphere+j]/4.))) * scheme.weights[j]
    Qb[i] += 1./6 * (nabla2rhob[i*nsphere+j] - 2. * (2*taua[i*nsphere+j]-(np.sum(rho1b[i*nsphere+j]**2+rho2b[i*nsphere+j]**2+rho3b[i*nsphere+j]**2)/rhob[i*nsphere+j]/4.))) * scheme.weights[j]
    density_a[i] += np.sqrt(np.sum(rhoa[i*nsphere+j]**2)) * scheme.weights[j]
    density_b[i] += np.sqrt(np.sum(rhob[i*nsphere+j]**2)) * scheme.weights[j]

  if np.isnan(Qa[i]): y_a[i] = 0
  else: y_a[i] += 2./3 * np.pi**(2./3) * density_a[i]**(5./3) / Qa[i]
  if np.isnan(Qb[i]): y_b[i] = 0
  else: y_b[i] += 2./3 * np.pi**(2./3) * density_b[i]**(5./3) / Qb[i]

Smat = np.zeros((nb,nb))
taumat = np.zeros((nb,nb))
Pmat = np.zeros((nb,nb))

results = brholegrid(x, w, scheme.azimuthal_polar, scheme.weights, y_a, y_b, density_a, density_b, 0,0,0,0,0,0)
print(len(x), len(scheme.weights), results[2])
time0 = time.time()
for i in range(nb):
  for j in range(nb):
    if (bbasis[i,0]-bbasis[j,0])%2==0 and (bbasis[i,1]-bbasis[j,1])%2==0 and (bbasis[i,2]-bbasis[j,2])%2==0:
      results = brholegrid(x, w, scheme.azimuthal_polar, scheme.weights, y_a, y_b, density_a, density_b, bbasis[i,0], bbasis[i,1], bbasis[i,2], bbasis[j,0], bbasis[j,1], bbasis[j,2])
      Smat[i,j] = results[0]
      taumat[i,j] = results[1]
      Pmat[i,j] = results[2]
time1 = time.time()

print("Total time: %.3f   Time per integral: %.3f" % (time1-time0, (time1-time0)/nb**2))

# Take a symmetric average to correct to the nonexact behaviour of the approximate exchange-hole
Pmat = (Pmat + Pmat.T)/2.
# Extract dipole moment terms from the overlap matrices
dlist = np.zeros((nb,3))
for i in range(nb):
  for j in range(nb):
    if bbasis[i,0] == 1 and bbasis[i,1] == 0 and  bbasis[i,2] == 0:
      dlist[j,0] += Smat[i,j] + Pmat[i, j]
    elif bbasis[i,1] == 1 and bbasis[i,0] == 0 and  bbasis[i,2] == 0:
      dlist[j,1] += Smat[i,j] + Pmat[i, j]
    elif bbasis[i,2] == 1 and bbasis[i,0] == 0 and  bbasis[i,1] == 0:
      dlist[j,2] += Smat[i,j] + Pmat[i, j]

# Compute the C6 coefficient
from scipy.linalg import eigh

taulist, tauvecs = eigh(taumat, Smat + Pmat)
dlist = tauvecs.T.dot(dlist)
dlist_sq = np.sum(dlist**2, axis=1)
C6 = 0
for i in range(nb):
  for j in range(nb):
    C6 += dlist_sq[i]*dlist_sq[j] / (taulist[i] + taulist[j]) * 4./3
print("BR hole C6: ", C6)
