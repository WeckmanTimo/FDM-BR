from distutils.core import setup, Extension
import sysconfig

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-lm",  "-fopenmp", "-I/home/userdir/anaconda3/lib/python3.7/site-packages/numpy/core/include/"]

def main():
    setup(name="xhole",
          version="1.0.0",
          description="Python interface to PySCF for computing the Becke-Roussel multipole moments for FDM dispersion coefficients.",
          author="Timo Weckman",
          author_email="timo.weckman@gmail.com",
          ext_modules=[Extension("xhole", ["BR-hole.c"], extra_compile_args=extra_compile_args, extra_link_args=extra_compile_args)])
    setup(name="HFpqrs",
          version="1.0.0",
          description="Python interface to PySCF for computing the exchange energy correction for FDM dispersion coefficients when using Hartree-Fock pairdensity.",
          author="Timo Weckman",
          author_email="timo.weckman@gmail.com",
          ext_modules=[Extension("HFpqrs", ["HF-pqrs.c"], extra_compile_args=extra_compile_args, extra_link_args=extra_compile_args)])
if __name__ == "__main__":
    main()
