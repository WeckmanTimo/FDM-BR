#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#define pi 3.141592653589793

#define rhoa(x0) (*(npy_float64*)((PyArray_DATA(py_rhoa) +              \
                                    (x0) * PyArray_STRIDES(py_rhoa)[0])))
#define rhoa_shape(i) (py_rhoa->dimensions[(i)])

#define rhob(x0) (*(npy_float64*)((PyArray_DATA(py_rhob) +              \
                                    (x0) * PyArray_STRIDES(py_rhob)[0])))
#define rhob_shape(i) (py_rhob->dimensions[(i)])

#define bopt1_fit(x0) (*(npy_float64*)((PyArray_DATA(py_bopt1) +              \
                                    (x0) * PyArray_STRIDES(py_bopt1)[0])))
#define bopt1_fit_shape(i) (py_bopt1->dimensions[(i)])

#define bopt2_fit(x0) (*(npy_float64*)((PyArray_DATA(py_bopt2) +              \
                                    (x0) * PyArray_STRIDES(py_bopt2)[0])))
#define bopt2_fit_shape(i) (py_bopt2->dimensions[(i)])

#define grid(x0) (*(npy_float64*)((PyArray_DATA(py_gl) +              \
                                    (x0) * PyArray_STRIDES(py_gl)[0])))
#define grid_shape(i) (py_gl->dimensions[(i)])

#define grid_weights(x0) (*(npy_float64*)((PyArray_DATA(py_glw) +              \
                                    (x0) * PyArray_STRIDES(py_glw)[0])))
#define grid_weights_shape(i) (py_glw->dimensions[(i)])

#define lebedev(x0, x1) (*(npy_float64*)((PyArray_DATA(py_leb) +              \
                                    (x0) * PyArray_STRIDES(py_leb)[0] + \
                                    (x1) * PyArray_STRIDES(py_leb)[1])))
#define lebedev_shape(i) (py_leb->dimensions[(i)])

#define lebedev_weights(x0) (*(npy_float64*)((PyArray_DATA(py_lebw) +              \
                                    (x0) * PyArray_STRIDES(py_lebw)[0])))
#define lebedev_weights_shape(i) (py_lebw->dimensions[(i)])

#define psi(x0, x1) (*(npy_float64*)((PyArray_DATA(py_psi) +              \
                                    (x0) * PyArray_STRIDES(py_psi)[0] + \
                                    (x1) * PyArray_STRIDES(py_psi)[1])))
#define psi_shape(i) (py_psi->dimensions[(i)])

#define gamma(x0, x1, x2, x3) (*(npy_float64*)((PyArray_DATA(py_gamma) +              \
                                    (x0) * PyArray_STRIDES(py_gamma)[0] + \
                                    (x1) * PyArray_STRIDES(py_gamma)[1] + \
                                    (x2) * PyArray_STRIDES(py_gamma)[2] + \
                                    (x3) * PyArray_STRIDES(py_gamma)[3])))
#define gamma_shape(i) (py_gamma->dimensions[(i)])

#define rdm1a(x0, x1) (*(npy_float64*)((PyArray_DATA(py_rdm1a) +              \
                                    (x0) * PyArray_STRIDES(py_rdm1a)[0] + \
                                    (x1) * PyArray_STRIDES(py_rdm1a)[1])))
#define rdm1a_shape(i) (py_rdm1a->dimensions[(i)])

#define rdm1b(x0, x1) (*(npy_float64*)((PyArray_DATA(py_rdm1b) +              \
                                    (x0) * PyArray_STRIDES(py_rdm1b)[0] + \
                                    (x1) * PyArray_STRIDES(py_rdm1b)[1])))
#define rdm1b_shape(i) (py_rdm1b->dimensions[(i)])

#define results(x0) (*(npy_float64*)((PyArray_DATA(py_results) +              \
                                    (x0) * PyArray_STRIDES(py_results)[0])))
#define results_shape(i) (py_results->dimensions[(i)])

#define cart_grid(x0, x1) (*(npy_float64*)((PyArray_DATA(py_cart_gl) +              \
                                    (x0) * PyArray_STRIDES(py_cart_gl)[0] + \
                                    (x1) * PyArray_STRIDES(py_cart_gl)[1])))
#define cart_grid_shape(i) (py_cart_gl->dimensions[(i)])

#define cart_grid_weights(x0) (*(npy_float64*)((PyArray_DATA(py_cart_glw) +              \
                                    (x0) * PyArray_STRIDES(py_cart_glw)[0])))
#define cart_grid_weights_shape(i) (py_cart_glw->dimensions[(i)])



// Forward function declaration
static PyObject * FDM_integrals(PyObject *self, PyObject *args);

static PyMethodDef FDMMethods[] = {
    {"HFpqrs", FDM_integrals, METH_VARARGS, "Python C API for computing FDM dispersion interaction integrals"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef FDMmodule = {
    PyModuleDef_HEAD_INIT,
    "HFpqrs",
    "Python interface for computing FDM dispersion interaction integrals",
    -1,
    FDMMethods
};

PyMODINIT_FUNC PyInit_HFpqrs(void) {
    import_array(); // Required for NumPy
    return PyModule_Create(&FDMmodule);
}

double exchange_integral(double psip, double psir, double psiq, double psis, double u){
  return psip * psir * psiq * psis / u;
}

double Kintegral(double psip, double psir, double psiq, double psis, double bi1, double bk1, double bi2, double bk2, double u){
  return psip * psir * psiq * psis / u * (bi1*bk1 + bi2*bk2 - bi1*bk2 - bi2*bk1);
}

static PyObject * FDM_integrals(PyObject *self, PyObject *args){
  int i,j,k,l,p,q,r,s;
  double integ, u;

  PyArrayObject * py_gl;
  PyArrayObject * py_glw;
  PyArrayObject * py_leb;
  PyArrayObject * py_lebw;
  PyArrayObject * py_psi;
  PyArrayObject * py_rdm1a;
  PyArrayObject * py_rdm1b;
  PyArrayObject * py_gamma;
  PyArrayObject * py_rhoa;
  PyArrayObject * py_rhob;
  PyArrayObject * py_bopt1;
  PyArrayObject * py_bopt2;
  PyArrayObject * py_results;

  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!O!O!",
    &PyArray_Type, &py_gl,
    &PyArray_Type, &py_glw,
    &PyArray_Type, &py_leb,
    &PyArray_Type, &py_lebw,
    &PyArray_Type, &py_psi,
    &PyArray_Type, &py_rdm1a,
    &PyArray_Type, &py_rdm1b,
    &PyArray_Type, &py_gamma,
    &PyArray_Type, &py_rhoa,
    &PyArray_Type, &py_rhob,
    &PyArray_Type, &py_bopt1,
    &PyArray_Type, &py_bopt2,
    &PyArray_Type, &py_results
    )){return NULL;}

  double * bopt1 = (double *) malloc((grid_shape(0)) * sizeof(double ));
  double * bopt2 = (double *) malloc((grid_shape(0)) * sizeof(double ));
  double * nablabopt1 = (double *) malloc((grid_shape(0)) * sizeof(double ));
  double * nablabopt2 = (double *) malloc((grid_shape(0)) * sizeof(double ));

  for(i=0;i<grid_shape(0);i++){ bopt1[i] = 0; nablabopt1[i] = 0; bopt2[i] = 0; nablabopt2[i] = 0;}

  // Initialize b-functions
  for(i=0;i<grid_shape(0);i++){
    for(j=0; j< bopt1_fit_shape(0);j++){
      bopt1[i] += bopt1_fit(j) * pow(grid(i), j);
      nablabopt1[i] += bopt1_fit(j) * (j * pow(grid(i), j-1) );
    }
    for(j=0; j< bopt2_fit_shape(0);j++){
      bopt2[i] += bopt2_fit(j) * pow(grid(i), j);
      nablabopt2[i] += bopt2_fit(j) * (j * pow(grid(i), j-1) );
    }
  }

  double r1sq, r2sq;
  double di, tauij, Sij, Pij, Di, Kcorrect, Kinteg;
  di = tauij = Sij = Pij = Di = Kcorrect = 0.;
  #pragma omp parallel for default(shared) private(p,q,r,s,i,j,k,l,u,r1sq,r2sq,integ,Kinteg) reduction(+: di) reduction(+: tauij) reduction(+: Sij) reduction(+: Di) reduction(+: Pij) reduction(+: Kcorrect)
  for (i=0; i< (int) grid_shape(0);i++){
    r1sq = pow(grid(i), 2);
    for(k=0; k< (int) lebedev_shape(0); k++){
      di += 4*pi * bopt1[i] * pow(grid(i), 3) * (rhoa(i) + rhob(i)) * grid_weights(i) * pow(cos(lebedev(k,1)), 2) * lebedev_weights(k);
      tauij += 4*pi * (rhoa(i) + rhob(i)) * (nablabopt1[i] * nablabopt2[i] * pow(grid(i), 2) + 2 * bopt1[i] * bopt2[i]) * grid_weights(i) * pow(cos(lebedev(k,1)), 2) * lebedev_weights(k);
      Sij += 4*pi * (rhoa(i) + rhob(i)) * bopt1[i] * bopt2[i] * r1sq * grid_weights(i) * pow(cos(lebedev(k,1)), 2) * lebedev_weights(k);
      for (j=0; j<grid_shape(0);j++){
        r2sq = pow(grid(j), 2);
        for(l=0; l< (int) lebedev_shape(0); l++){
          u = sqrt(r1sq + r2sq - 2 * sqrt(r1sq*r2sq) * (sin(lebedev(k,1)) * sin(lebedev(l,1)) * cos(lebedev(k,0) - lebedev(l,0)) + cos(lebedev(k,1)) * cos(lebedev(l,1))) );
          for(p=0; p< (int) rdm1a_shape(0); p++){
            for(q=0; q< (int) rdm1b_shape(0); q++){
              for(r=0; r< (int) rdm1a_shape(1); r++){ // r=p
                for(s=0; s< (int) rdm1b_shape(1); s++){ // s=q
                  // 2-RDM integral for Pij and Di
                  integ = gamma(p,r,q,s) * psi(i*lebedev_shape(0) + k,p) * psi(i*lebedev_shape(0) + k,r) * psi(j*lebedev_shape(0) + l,q) * psi(j*lebedev_shape(0) + l,s) * r1sq * r2sq * grid_weights(i) * grid_weights(j) * cos(lebedev(k,1)) * cos(lebedev(l,1));
                  Pij += integ * bopt1[i] * bopt2[j] * lebedev_weights(k) * lebedev_weights(l) * pow(4*pi,2);
                  Di += integ * grid(i) * bopt2[j]* lebedev_weights(k) * lebedev_weights(l) * pow(4*pi,2);
                  // K-correction integral with AO
                  if(u>1e-12){Kinteg = (rdm1a(p,q)*rdm1a(r,s) + rdm1b(p,q)*rdm1b(r,s)) * Kintegral(psi(i*lebedev_shape(0) + k,p), psi(i*lebedev_shape(0) + k,r), psi(j*lebedev_shape(0) + l,q), psi(j*lebedev_shape(0) + l,s), bopt1[i]*cos(lebedev(k,1)), bopt2[i]*cos(lebedev(k,1)), bopt1[j]*cos(lebedev(l,1)), bopt2[j]*cos(lebedev(l,1)), u) * r1sq * r2sq * grid_weights(i) * grid_weights(j) * lebedev_weights(k) * lebedev_weights(l) * pow(4*pi,2); Kcorrect += Kinteg;}
                }
              }
            }
          }
        }
      }
   }
  }
  results(0) = di;
  results(1) = tauij;
  results(2) = Sij;
  results(3) = Pij;
  results(4) = Di;
  results(5) = Kcorrect;
  return  PyFloat_FromDouble(Pij);
}
