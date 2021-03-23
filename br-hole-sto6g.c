#include <Python.h>
#include <numpy/arrayobject.h>
#include <arb.h>
#include <arb_hypgeom.h>
#include <mag.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#define pi 3.141592653589793

#define gauss_legendre(x0) (*(npy_float64*)((PyArray_DATA(py_gl) +              \
                                    (x0) * PyArray_STRIDES(py_gl)[0])))
#define gauss_legendre_shape(i) (py_gl->dimensions[(i)])

#define gauss_legendre_weights(x0) (*(npy_float64*)((PyArray_DATA(py_glw) +              \
                                    (x0) * PyArray_STRIDES(py_glw)[0])))
#define gauss_legendre_weights_shape(i) (py_glw->dimensions[(i)])

#define y_a(x0) (*(npy_float64*)((PyArray_DATA(py_y_a) +              \
                                    (x0) * PyArray_STRIDES(py_y_a)[0])))
#define y_a_shape(i) (py_y_a->dimensions[(i)])

#define y_b(x0) (*(npy_float64*)((PyArray_DATA(py_y_b) +              \
                                    (x0) * PyArray_STRIDES(py_y_b)[0])))
#define y_b_shape(i) (py_y_b->dimensions[(i)])

#define lebedev(x0, x1) (*(npy_float64*)((PyArray_DATA(py_leb) +              \
                                    (x0) * PyArray_STRIDES(py_leb)[0] + \
                                    (x1) * PyArray_STRIDES(py_leb)[1])))
#define lebedev_shape(i) (py_leb->dimensions[(i)])

#define lebedev_weights(x0) (*(npy_float64*)((PyArray_DATA(py_lebw) +              \
                                    (x0) * PyArray_STRIDES(py_lebw)[0])))
#define lebedev_weights_shape(i) (py_lebw->dimensions[(i)])

#define rhoa(x0) (*(npy_float64*)((PyArray_DATA(py_rhoa) +              \
                                    (x0) * PyArray_STRIDES(py_rhoa)[0])))
#define rhoa_shape(i) (py_rhoa->dimensions[(i)])

#define rhob(x0) (*(npy_float64*)((PyArray_DATA(py_rhob) +              \
                                    (x0) * PyArray_STRIDES(py_rhob)[0])))
#define rhob_shape(i) (py_rhob->dimensions[(i)])

#define results(x0) (*(npy_float64*)((PyArray_DATA(py_results) +              \
                                    (x0) * PyArray_STRIDES(py_results)[0])))
#define results_shape(i) (py_results->dimensions[(i)])

// Forward function declaration
static PyObject * brhole_cart(PyObject *self, PyObject *args);
static PyObject * brhole_sto6g_cart(PyObject *self, PyObject *args);

static PyMethodDef BRMethods[] = {
    {"brholegrid", brhole_cart, METH_VARARGS, "Python interface for computing Becke--Roussel multipole moment integrals on a grid"},
	{"brholesto6g", brhole_sto6g_cart, METH_VARARGS, "Python interface for computing Becke--Roussel multipole moment integrals on a grid using an STO-6G approximation"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef brmodule = {
    PyModuleDef_HEAD_INIT,
    "brhole",
    "Python interface for the multipole integrals for the FDM procedure using the Becke--Roussel exchange-hole",
    -1,
    BRMethods
};

PyMODINIT_FUNC PyInit_brhole(void) {
    import_array();
    return PyModule_Create(&brmodule);
}

long double exponential(double x){
  long prec = 64;
  mag_t res_mag; mag_init(res_mag);
  arb_t exp_arg; arb_init(exp_arg);
  arb_t exp_res; arb_init(exp_res);
  arb_set_d(exp_arg, x);
  arb_exp(exp_res, exp_arg, prec);
  arb_get_mag(res_mag, exp_res);
  long double result;
  result = mag_get_d(res_mag);
  if(isnan(result)){printf("Exp NaN with arg %lf\n", x);}
  return result;
}

long double arccsch(long double x){
  return log(1/x + sqrt(1/pow(x,2) + 1));
}

long double x_sigma(long double y){
/* 
*  Analytical expression for the Becke--Roussel exchange hole, obtained from 
*  Proynov, Emil, Zhenting Gan, and Jing Kong. "Analytical representation 
*  of the Becke–Roussel exchange functional." Chemical Physics Letters 455.1-3 (2008): 103-109.
*/
  static const double A1 = 1.52552518120095; static const double A2 = 0.457657554360286; 
  static const double A3 = 0.429203673205103; static const double B = 2.08574971649376; 
  static const double B0 = 0.4771976184; static const double B1 = -1.7799813495;
  static const double B2 = 3.8433841862; static const double B3 = -9.5912050881;
  static const double B4 = 2.1730180286; static const double B5 = -30.4251338516037;
  static const double C0 = 0.7566445421; static const double C1 = -2.6363977871;
  static const double C2 = 5.4745159964; static const double C3 = -12.6573081271; 
  static const double C4 = 4.1250584725; static const double C5 = -30.4251339572; 
  static const double D0 = 4.43500988679559e-5; static const double D1 = 0.581286536;
  static const double D2 = 66.7427645159; static const double D3 = 434.2678089723; 
  static const double D4 = 824.7765766052; static const double D5 = 1657.9652731582; 
  static const double E0 = 3.34728506092609e-5; static const double E1 = 0.4791793102;
  static const double E2 = 62.3922683386; static const double E3 = 463.1481642794; 
  static const double E4 = 785.2360350104; static const double E5 = 1657.96296822327; 
  
  if(fabs(y) <1e-50){return 0;}
  else if (y<0){return (A3 - atan(A1*y + A2))*(C5*pow(y,5) + C4*pow(y,4) + C3*pow(y,3) + C2*pow(y,2) + C1*y + C0)/(B5*pow(y,5) + B4*pow(y,4) + B3*pow(y,3) + B2*pow(y,2) + B1*y + B0);}
  else if (y>0){return (arccsch(B*y) + 2)*(D5*pow(y,5) + D4*pow(y,4) + D3*pow(y,3) + D2*pow(y,2) + D1*y + D0)/(E5*pow(y,5) + E4*pow(y,4) + E3*pow(y,3) + E2*pow(y,2) + E1*y + E0);}
  return 0;
}

long double BRhole_sph_arb(double a_a, double coeff_a, double a_b, double coeff_b, double rhoa, double rhob, double r1, double theta1, double phi1, double r2, double theta2, double phi2, double s1, double t1, double u1, double s2, double t2, double u2){
  double holea, holeb;
  holea = rhoa * exponential(-a_a*sqrt(fabs( pow(coeff_a*r1,2) + pow(r2,2) + 2*coeff_a*r1*r2 * (sin(theta1) * sin(theta2) * cos(phi1-phi2) + cos(theta1)*cos(theta2) ))));
  holeb = rhob * exponential(-a_b*sqrt(fabs( pow(coeff_b*r1,2) + pow(r2,2) + 2*coeff_b*r1*r2 * (sin(theta1) * sin(theta2) * cos(phi1-phi2) + cos(theta1)*cos(theta2) ))));
  return -(pow(a_a,3) / (8*pi) * holea + pow(a_b,3) / (8*pi) * holeb) * pow(r1,2+s1+t1+u1) * pow(r2,2+s2+t2+u2) * pow(sin(theta1),s1+t1) * pow(cos(theta1),u1) * pow(sin(theta2),s2+t2) * pow(cos(theta2),u2) * pow(cos(phi1),s1) * pow(sin(phi1),t1) * pow(cos(phi2),s2) * pow(sin(phi2),t2);
}

double gaussian_integral(double n, double c, double a){
  long prec = 64;
  double result;
  mag_t x; mag_init(x);
  arb_t a_arb, b_arb, z_arb, res_arb, expac2, hypergeom, z_arb_neg;
  arb_init(a_arb); arb_init(b_arb); arb_init(z_arb); arb_init(res_arb); arb_init(expac2); arb_init(hypergeom); arb_init(z_arb_neg);
  if((int)n%2!=0){
    arb_set_d(a_arb, 1.+n/2); arb_set_d(b_arb, 3./2); arb_set_d(z_arb, pow(c,2)*a); arb_set_d(res_arb, 0);
    arb_hypgeom_1f1(hypergeom, a_arb, b_arb, z_arb, 0, prec); // 1F1
    arb_neg(z_arb_neg, z_arb);
    arb_exp(expac2, z_arb_neg, prec); // Exponential term
    arb_mul(res_arb, expac2, hypergeom, prec); // Product of 1F1 and exponential
    arb_get_mag(x, res_arb);
    result = mag_get_d(x);
    return -pow(a,-(1+n)/2.) * 2. * c * sqrt(a) * tgamma(1.+n/2.) * result;
  }
  else{
    arb_set_d(a_arb, (1.+n)/2); arb_set_d(b_arb, 1./2); arb_set_d(z_arb, pow(c,2)*a); arb_set_d(res_arb, 0);
    arb_hypgeom_1f1(hypergeom, a_arb, b_arb, z_arb, 0, prec); // 1F1
    arb_neg(z_arb_neg, z_arb);
    arb_exp(expac2, z_arb_neg, prec); // Exponential term
    arb_mul(res_arb, expac2, hypergeom, prec); // Product of 1F1 and exponential
    arb_get_mag(x, res_arb);
    result = mag_get_d(x);
    return pow(a,-(1+n)/2.) * tgamma((1.+n)/2) * result;
  }
}


double BRhole_sto6g_cart (double a, double coeff, double x1, double y1, double z1, double s1, double t1, double u1, double s2, double t2, double u2){
  double integral = 0;
// STO-3G parameters
  double sto_exp[3] = {2.22766, 0.405771, 0.109818};
  double sto_coeff[3] = {0.154329, 0.535328, 0.444635};
// STO-6G parameters
//  double sto_exp[6] = {0.0651095, 0.158088, 0.407099, 1.18506, 4.23592, 23.103};
//  double sto_coeff[6] = {0.130334, 0.416492, 0.370563, 0.168538, 0.0493615, 0.0091636};

// Approximate BR-hole with STO-3G or STO-6G function
// Analytical integral over x_2 coordinates, separates into xyz-product
  for(int sto1=0; sto1<3; sto1++){
    for(int sto2=0; sto2<3; sto2++){
      integral -= sto_coeff[sto1] * sto_coeff[sto2] * pow(2 * pow(a,2) * sto_exp[sto1] / pi, 3./4) * pow(2 * pow(a,2) * sto_exp[sto2] / pi, 3./4) * pow(x1,s1) * gaussian_integral(s2, coeff * x1, pow(a,2) * (sto_exp[sto1] + sto_exp[sto2]) / 4.) * pow(y1,t1) * gaussian_integral(t2, coeff * y1, pow(a,2) * (sto_exp[sto1] + sto_exp[sto2])/ 4.) * pow(z1,u1) * gaussian_integral(u2, coeff * z1, pow(a,2) * (sto_exp[sto1] + sto_exp[sto2]) / 4.) / 8;
    }
  }
  return integral;
}

static PyObject * brhole_cart(PyObject *self, PyObject *args){
/*
*  This function computes multipole moments for the Fixed Diagonal Matrices (FDM)
*  procedure using the Becke--Roussel (BR) exchange hole for the pair-density mediated 
*  terms. The integrals are done using a numerical grid for the radial part and a 
*  Lebedev-Laikov grid for the angular parts.
*/
  int i,j,k,l;
  double a_a, a_b, b_a, b_b, x, coeff_a, coeff_b, integ;
  double s1, t1, u1, s2, t2 ,u2;
  PyArrayObject * py_gl;
  PyArrayObject * py_glw;
  PyArrayObject * py_leb;
  PyArrayObject * py_lebw;
  PyArrayObject * py_y_a;
  PyArrayObject * py_y_b;
  PyArrayObject * py_rhoa;
  PyArrayObject * py_rhob;
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!dddddd",
    &PyArray_Type, &py_gl, // A nonlinear radial grid of type Gauss-Legendre or Gauss-Laguerre, dimensions (M,)
    &PyArray_Type, &py_glw, // Weights for the radial grid, dimensions (M,)
    &PyArray_Type, &py_leb, // Lebedev-Laikov grid, dimensions (N,2) where phi on axis=0 and theta on axis=1
    &PyArray_Type, &py_lebw, // Weights for the Lebedev-Laikov grid, dimensions (N,)
    &PyArray_Type, &py_y_a, // y_a on a grid, dimensions (N,)
    &PyArray_Type, &py_y_b, // y_b on a grid, dimensions (N,)
    &PyArray_Type, &py_rhoa, // Spherically averaged spin-density of spin-A on the same grid as py_gl, (M,)
    &PyArray_Type, &py_rhob, // Spherically averaged spin-density of spin-B on the same grid as py_gl, (M,)
    &s1, &t1, &u1, &s2, &t2, &u2 // Monomial exponents, x^s * y^t * z^u for electrons 1 and 2
     )){return NULL;}

  double tauij, Sij, Pij;
  tauij = Sij = Pij = 0;

  for (i=0; i< (int) gauss_legendre_shape(0);i++){

    x = x_sigma(y_a(i));
    b_a = pow(pow(x,3) * exponential(-x) / (8 * pi * rhoa(i)), 1./3.);
    if(b_a==0 || isnan(b_a)){a_a = 0; coeff_a = -1;}
    else{a_a = x / b_a; coeff_a = b_a / gauss_legendre(i) - 1;}

    x = x_sigma(y_b(i));
    b_b = pow(pow(x,3) * exponential(-x) / (8 * pi * rhob(i)), 1./3.);
    if(b_b==0 || isnan(b_b)){a_b = 0; coeff_b = -1;}
    else{a_b = x / b_b; coeff_b = b_b / gauss_legendre(i) - 1;}

    #pragma omp parallel for default(shared) private(j,k,l,integ) reduction(+: Pij) reduction(+: tauij) reduction(+: Sij) 
    for (j=0; j< lebedev_shape(0);j++){
		// Numerical integration for the kinetic energy terms, tauij
        if(s1+s2>0){tauij += 4*pi * (rhoa(i) + rhob(i)) * s1*s2 * pow(gauss_legendre(i),s1+s2+t1+t2+u1+u2) * pow(sin(lebedev(j,1)),s1+s2+t1+t2-2) * pow(cos(lebedev(j,1)),u1+u2) * pow(cos(lebedev(j,0)),s1+s2-2) * pow(sin(lebedev(j,0)),t1+t2) * gauss_legendre_weights(i) * lebedev_weights(j);}
        if(t1+t2>0){tauij += 4*pi * (rhoa(i) + rhob(i)) * t1*t2 * pow(gauss_legendre(i),s1+s2+t1+t2+u1+u2) * pow(sin(lebedev(j,1)),s1+s2+t1+t2-2) * pow(cos(lebedev(j,1)),u1+u2) * pow(cos(lebedev(j,0)),s1+s2) * pow(sin(lebedev(j,0)),t1+t2-2) * gauss_legendre_weights(i) * lebedev_weights(j);}
        if(u1+u2>0){tauij += 4*pi * (rhoa(i) + rhob(i)) * u1*u2 *pow(gauss_legendre(i),s1+s2+t1+t2+u1+u2) * pow(sin(lebedev(j,1)),s1+s2+t1+t2) * pow(cos(lebedev(j,1)),u1+u2-2) * pow(cos(lebedev(j,0)),s1+s2) * pow(sin(lebedev(j,0)),t1+t2) * gauss_legendre_weights(i) * lebedev_weights(j);}
		// Numerical integration for the 1-electron density overlap, Sij
        Sij += 4*pi * (rhoa(i) + rhob(i)) * pow(gauss_legendre(i),2+s1+s2+t1+t2+u1+u2) * pow(sin(lebedev(j,1)),s1+t1+s2+t2) * pow(cos(lebedev(j,1)),u1+u2) * pow(cos(lebedev(j,0)),s1+s2) * pow(sin(lebedev(j,0)),t1+t2) * gauss_legendre_weights(i) * lebedev_weights(j);
		// Numerical integration for the pair-density mediated overlap, Pij
        for (k=0; k< gauss_legendre_shape(0);k++){
			for (l=0; l< lebedev_shape(0);l++){
				integ = pow(4*pi,2) * BRhole_sph_arb(a_a, coeff_a, a_b, coeff_b, rhoa(i), rhob(i), gauss_legendre(i), lebedev(j,1), lebedev(j,0), gauss_legendre(k), lebedev(l,1), lebedev(l,0), s1, t1, u1, s2, t2, u2);
				Pij += integ * gauss_legendre_weights(i) * lebedev_weights(j) * gauss_legendre_weights(k) * lebedev_weights(l);
			}
		}
    }
  }
  double results_array[3] = {Sij, tauij, Pij};
  npy_intp dims[1] = {3};
  PyObject *output = PyArray_SimpleNew(1,dims, NPY_DOUBLE);
  memcpy(PyArray_DATA(output), results_array, sizeof(results_array));
  return output;
}

static PyObject * brhole_sto6g_cart(PyObject *self, PyObject *args){
/*
*  This function computes multipole moments for the Fixed Diagonal Matrices (FDM)
*  procedure using the Becke--Roussel (BR) exchange hole for the pair-density mediated 
*  terms. The BR-hole integrals are approximated using a STO-nG-type approximation.
*  The exponential form of the BR-hole is approximated using an STO-3G (or STO-6G) type
*  function. This allows for an analytical integration over the second electron.
*/
  int i,j;
  double a_a, a_b, b_a, b_b, x, coeff_a, coeff_b, integ;
  double x1, y1, z1;
  double s1, t1, u1, s2, t2 ,u2;
  PyArrayObject * py_gl;
  PyArrayObject * py_glw;
  PyArrayObject * py_leb;
  PyArrayObject * py_lebw;
  PyArrayObject * py_y_a;
  PyArrayObject * py_y_b;
  PyArrayObject * py_rhoa;
  PyArrayObject * py_rhob;
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!dddddd",
    &PyArray_Type, &py_gl, // A nonlinear radial grid of type Gauss-Legendre or Gauss-Laguerre, dimensions (M,)
    &PyArray_Type, &py_glw, // Weights for the radial grid, dimensions (M,)
    &PyArray_Type, &py_leb, // Lebedev-Laikov grid, dimensions (N,2) where phi on axis=0 and theta on axis=1
    &PyArray_Type, &py_lebw, // Weights for the Lebedev-Laikov grid, dimensions (N,)
    &PyArray_Type, &py_y_a, // y_a on a grid, dimensions (N,)
    &PyArray_Type, &py_y_b, // y_b on a grid, dimensions (N,)
    &PyArray_Type, &py_rhoa, // Spherically averaged spin-density of spin-A on the same grid as py_gl, (M,)
    &PyArray_Type, &py_rhob, // Spherically averaged spin-density of spin-B on the same grid as py_gl, (M,)
    &s1, &t1, &u1, &s2, &t2, &u2 // Monomial exponents, x^s * y^t * z^u for electrons 1 and 2
     )){return NULL;}

  double tauij, Sij, Pij;
  tauij = Sij = Pij = 0;

  for (i=0; i< (int) gauss_legendre_shape(0);i++){

    x = x_sigma(y_a(i));
    b_a = pow(pow(x,3) * exponential(-x) / (8 * pi * rhoa(i)), 1./3.);
    if(b_a==0 || isnan(b_a)){a_a = 0; coeff_a = -1;}
    else{a_a = x / b_a; coeff_a = b_a / gauss_legendre(i) - 1;}

    x = x_sigma(y_b(i));
    b_b = pow(pow(x,3) * exponential(-x) / (8 * pi * rhob(i)), 1./3.);
    if(b_b==0 || isnan(b_b)){a_b = 0; coeff_b = -1;}
    else{a_b = x / b_b; coeff_b = b_b / gauss_legendre(i) - 1;}

    #pragma omp parallel for default(shared) private(j, x1, y1, z1,integ) reduction(+: Pij) reduction(+: tauij) reduction(+: Sij) 
    for (j=0; j< lebedev_shape(0);j++){
		// Numerical integration for the kinetic energy terms, tauij
        if(s1+s2>0){tauij += 4*pi * (rhoa(i) + rhob(i)) * s1*s2 * pow(gauss_legendre(i),s1+s2+t1+t2+u1+u2) * pow(sin(lebedev(j,1)),s1+s2+t1+t2-2) * pow(cos(lebedev(j,1)),u1+u2) * pow(cos(lebedev(j,0)),s1+s2-2) * pow(sin(lebedev(j,0)),t1+t2) * gauss_legendre_weights(i) * lebedev_weights(j);}
        if(t1+t2>0){tauij += 4*pi * (rhoa(i) + rhob(i)) * t1*t2 * pow(gauss_legendre(i),s1+s2+t1+t2+u1+u2) * pow(sin(lebedev(j,1)),s1+s2+t1+t2-2) * pow(cos(lebedev(j,1)),u1+u2) * pow(cos(lebedev(j,0)),s1+s2) * pow(sin(lebedev(j,0)),t1+t2-2) * gauss_legendre_weights(i) * lebedev_weights(j);}
        if(u1+u2>0){tauij += 4*pi * (rhoa(i) + rhob(i)) * u1*u2 *pow(gauss_legendre(i),s1+s2+t1+t2+u1+u2) * pow(sin(lebedev(j,1)),s1+s2+t1+t2) * pow(cos(lebedev(j,1)),u1+u2-2) * pow(cos(lebedev(j,0)),s1+s2) * pow(sin(lebedev(j,0)),t1+t2) * gauss_legendre_weights(i) * lebedev_weights(j);}
		// Numerical integration for the 1-electron density overlap, Sij
        Sij += 4*pi * (rhoa(i) + rhob(i)) * pow(gauss_legendre(i),2+s1+s2+t1+t2+u1+u2) * pow(sin(lebedev(j,1)),s1+t1+s2+t2) * pow(cos(lebedev(j,1)),u1+u2) * pow(cos(lebedev(j,0)),s1+s2) * pow(sin(lebedev(j,0)),t1+t2) * gauss_legendre_weights(i) * lebedev_weights(j);
		// Numerical integration for the pair-density mediated overlap, Pij
		x1 = gauss_legendre(i)*sin(lebedev(j,1))*cos(lebedev(j,0));
		y1 = gauss_legendre(i)*sin(lebedev(j,1))*sin(lebedev(j,0));
		z1 = gauss_legendre(i)*cos(lebedev(j,1));
		integ = 4*pi * (rhoa(i) * BRhole_sto6g_cart(a_a, coeff_a, x1, y1, z1, s1, t1, u1, s2, t2, u2) + rhob(i) * BRhole_sto6g_cart(a_b, coeff_b, x1, y1, z1, s1, t1, u1, s2, t2, u2)) * pow(gauss_legendre(i),2);
		Pij += integ * gauss_legendre_weights(i) * lebedev_weights(j);
    }
  }
  double results_array[3] = {Sij, tauij, Pij};
  npy_intp dims[1] = {3};
  PyObject *output = PyArray_SimpleNew(1,dims, NPY_DOUBLE);
  memcpy(PyArray_DATA(output), results_array, sizeof(results_array));
  return output;
}
