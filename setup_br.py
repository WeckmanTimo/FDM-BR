from distutils.core import setup, Extension
import sysconfig

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-larb", "-lm", "-fopenmp"]

def main():
    setup(name="HFpqrs",
          version="1.0.0",
          description="Python interface for PySCF for computing the exchange energy correction for FDM dispersion coefficients when using Hartree-Fock pairdensity.",
          author="Timo Weckman",
          author_email="timo.weckman@gmail.com",
          ext_modules=[Extension("HFpqrs", ["HF-pqrs.c"], extra_compile_args=extra_compile_args, extra_link_args=extra_compile_args)])
    setup(name="brhole",
          version="1.0.0",
          description="Python interface for PySCF for computing the Becke--Roussel exchange hole multipole moments using Cartesian monomials as b-functions.",
          author="Timo Weckman",
          author_email="timo.weckman@gmail.com",
          ext_modules=[Extension("brhole", ["br-hole-sto6g.c"], extra_compile_args=extra_compile_args, extra_link_args=extra_compile_args)])

if __name__ == "__main__":
    main()
