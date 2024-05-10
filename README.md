# Fixed diagonal matrices (FDM)

This program computes dispersion coefficients for atoms and molecules using the Fixed Diagonal Matrices (FDM) procedure (Kooi, Gori-Giorgi, J. Phys. Chem. Lett. 2019, 10, 1537). 

Install the modules with

python setup.py install

Archived calculations (the .npz-files in the .zip-files) can be analyzed by running the compute*.py-files. Extract the archives in a folder and run a compute*.py-file. The compute_br/exchange.py scripts print out the C6 coefficient of each atom with respect to the size of the monomial basis, while the compute_mixed*.py files print out the C6 coefficients for all the atom pairs in the highest monomial basis.

## FDM-BR
FDM procedure uses Becke-Roussel exchange-hole approximation and approximates the pair-density mediated terms (Di, Pij) using the Becke--Roussel exchange-hole. 
Run a Becke-Roussel exchange hole calculation with

python calculate_BR.py 2 He 0 0

## FDM-K

Exchange-correction to the FDM procedure introduces an exchange correction to when using the Hartree--Fock pair-density. 

Run an exchange-correction calculation with

python calculate_exchange.py 2 He HF 0 0
