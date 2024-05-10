#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#define pi 3.141592653589793

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

#define y_a(x0) (*(npy_float64*)((PyArray_DATA(py_y_a) +              \
                                    (x0) * PyArray_STRIDES(py_y_a)[0])))
#define y_a_shape(i) (py_y_a->dimensions[(i)])

#define y_b(x0) (*(npy_float64*)((PyArray_DATA(py_y_b) +              \
                                    (x0) * PyArray_STRIDES(py_y_b)[0])))
#define y_b_shape(i) (py_y_b->dimensions[(i)])

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

#define results(x0) (*(npy_float64*)((PyArray_DATA(py_results) +              \
                                    (x0) * PyArray_STRIDES(py_results)[0])))
#define results_shape(i) (py_results->dimensions[(i)])

// Forward function declaration
static PyObject * brhole_sph(PyObject *self, PyObject *args);

static PyMethodDef BRMethods[] = {
    {"sphbrhole", brhole_sph, METH_VARARGS, "Python C API for Becke & Roussel exchange hole multipole moments"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef brmodule = {
    PyModuleDef_HEAD_INIT,
    "xhole",
    "Python interface for computing dispersion interaction integrals using Becke-Roussel exchange-hole",
    -1,
    BRMethods
};

PyMODINIT_FUNC PyInit_xhole(void) {
    import_array(); // Required for NumPy
    return PyModule_Create(&brmodule);
}

double arccsch(double x){
  return log(1/x + sqrt(1/pow(x,2) + 1));
}

double x_sigma(double y){
  if(y<=0){return (0.429203673205103 - atan(1.52552518120095*y + 0.457657554360286))*(-30.4251339572*pow(y,5) + 4.1250584725*pow(y,4) - 12.6573081271*pow(y,3) + 5.4745159964*pow(y,2) - 2.6363977871*y + 0.7566445421)/(-30.4251338516037*pow(y,5) + 2.1730180286*pow(y,4) - 9.5912050881*pow(y,3) + 3.8433841862*pow(y,2) - 1.7799813495*y + 0.4771976184);}
  else{return (arccsch(2.08574971649376*y) + 2)*(1657.9652731582*pow(y,5) + 824.7765766052*pow(y,4) + 434.2678089723*pow(y,3) + 66.7427645159*pow(y,2) + 0.581286536*y + 4.43500988679559e-5)/(1657.96296822327*pow(y,5) + 785.2360350104*pow(y,4) + 463.1481642794*pow(y,3) + 62.3922683386*pow(y,2) + 0.4791793102*y + 3.34728506092609e-5);}
}

double BRhole_sph (double a_a, double coeff_a, double a_b, double coeff_b, double density_a, double density_b, double r1, double theta1, double phi1, double r2, double theta2, double phi2){
  double holea = 0;
  if(a_a>0){holea = exp(-a_a*sqrt( pow(coeff_a*r1,2) + pow(r2,2) + 2*coeff_a*r1*r2 * (sin(theta1) * sin(theta2) * cos(phi1-phi2) + cos(theta1)*cos(theta2) )));}
  double holeb = 0;
  if(a_b>0){holeb = exp(-a_b*sqrt( pow(coeff_b*r1,2) + pow(r2,2) + 2*coeff_b*r1*r2 * (sin(theta1) * sin(theta2) * cos(phi1-phi2) + cos(theta1)*cos(theta2) )));}
  return -(density_a * pow(a_a,3) / (8*pi) * holea + density_b * pow(a_b,3) / (8*pi) * holeb) * cos(theta1) * cos(theta2);
}

static PyObject * brhole_sph(PyObject *self, PyObject *args){
  int i,j,k,l;
  double a_a, a_b, b_a, b_b, x, coeff_a, coeff_b, integ;

  PyArrayObject * py_gl;
  PyArrayObject * py_glw;
  PyArrayObject * py_leb;
  PyArrayObject * py_lebw;
  PyArrayObject * py_y_a;
  PyArrayObject * py_y_b;
  PyArrayObject * py_rhoa;
  PyArrayObject * py_rhob;
  PyArrayObject * py_bopt1;
  PyArrayObject * py_bopt2;
  PyArrayObject * py_results;

  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!",
    &PyArray_Type, &py_gl, // Gauss-Legendre grid for radial part
    &PyArray_Type, &py_glw,
    &PyArray_Type, &py_leb, // Lebedev-Laikov grid for angular part
    &PyArray_Type, &py_lebw,
    &PyArray_Type, &py_y_a, // y_a on a grid for calculating Becke--Roussel parameters using Proynov parametrization
    &PyArray_Type, &py_y_b, // y_b on a grid
    &PyArray_Type, &py_rhoa, // density_a on a grid
    &PyArray_Type, &py_rhob, // density_b on a grid
    &PyArray_Type, &py_bopt1, // bi as a list of polynomial coefficients
    &PyArray_Type, &py_bopt2, // bj as a list of polynomial coefficients
    &PyArray_Type, &py_results //
    )){return NULL;}

  double * bopt1 = (double *) malloc((grid_shape(0)) * sizeof(double ));
  double * bopt2 = (double *) malloc((grid_shape(0)) * sizeof(double ));
  double * nablabopt1 = (double *) malloc((grid_shape(0)) * sizeof(double ));
  double * nablabopt2 = (double *) malloc((grid_shape(0)) * sizeof(double ));

  // Initialize the bopt-array
  for(i=0;i<grid_shape(0);i++){ bopt1[i] = 0; nablabopt1[i] = 0; bopt2[i] = 0; nablabopt2[i] = 0;}

  // Form the b-functions on a given grid using polynomial coefficients
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

  double di, tauij, Sij, Pij, Pij2, Di, Di2, test, r1sq, r2sq;
  di = tauij = Sij = Pij = Pij2 = Di = Di2 = test = 0;
  for (i=0; i< (int) grid_shape(0);i++){
    r1sq = pow(grid(i), 2);
    x = x_sigma(y_a(i));
    b_a = pow(pow(x,3) * exp(-x) / (8 * pi * rhoa(i)), 1./3.);
    if(b_a==0 || isnan(b_a)){a_a = 0; b_a = 0;}
    else{a_a = x / b_a;}
    coeff_a = b_a / grid(i) - 1;

    x = x_sigma(y_b(i));
    b_b = pow(pow(x,3) * exp(-x) / (8 * pi * rhob(i)), 1./3.);
    if(b_b==0 || isnan(b_b)){a_b = 0; b_b=0;}
    else{a_b = x / b_b;}
    coeff_b = b_b / grid(i) - 1;

    #pragma omp parallel for default(shared) private(j,k,l,integ,r2sq) reduction(+: di) reduction(+: Sij) reduction(+: tauij) reduction(+: Di) reduction(+: Di2) reduction(+: Pij) reduction(+: Pij2) reduction(+: test)
    for (k=0; k<lebedev_shape(0);k++){
      di += 4*pi * bopt1[i] * grid(i) * r1sq * (rhoa(i) + rhob(i)) * grid_weights(i) * pow(cos(lebedev(k,1)), 2) * lebedev_weights(k);
      tauij += 4*pi * (rhoa(i) + rhob(i)) * (nablabopt1[i] * nablabopt2[i] * r1sq + 2 * bopt1[i] * bopt2[i]) * grid_weights(i) * pow(cos(lebedev(k,1)), 2) * lebedev_weights(k);
      Sij += 4*pi * (rhoa(i) + rhob(i)) * bopt1[i] * bopt2[i] * r1sq * grid_weights(i) * pow(cos(lebedev(k,1)), 2) * lebedev_weights(k);
      for (j=0; j<grid_shape(0);j++){
        r2sq = pow(grid(j), 2);
        for (l=0; l<lebedev_shape(0);l++){
          integ = pow(4*pi,2) * BRhole_sph(a_a, coeff_a, a_b, coeff_b, rhoa(i), rhob(i), grid(i), lebedev(k,1), lebedev(k,0), grid(j), lebedev(l,1), lebedev(l,0)) * r1sq * r2sq * lebedev_weights(k) * grid_weights(j) * lebedev_weights(l) * grid_weights(i);
          Di += integ * bopt1[i] * grid(j);
          Di2 += integ * grid(i) * bopt2[j];
          Pij += integ * bopt1[i] * bopt2[j];
          Pij2 += integ * bopt1[j] * bopt2[i];
          test += integ / (cos(lebedev(k,1)) * cos(lebedev(l,1)));
        }
      }
    }
  }

  results(0) = di;
  results(1) = tauij;
  results(2) = Sij;
  results(3) = Di;
  results(4) = Di2;
  results(5) = Pij;
  results(6) = Pij2;

  return  PyFloat_FromDouble(test);
}
