/*
 * Library:  lmfit (Levenberg-Marquardt least squares fitting)
 *
 * File:     demo/curve1.c
 *
 * Contents: Example for curve fitting with lmcurve():
 *           fit a data set y(x) by a curve f(x;p).
 *
 * Note:     Any modification of this example should be copied to
 *           the manual page source lmcurve.pod and to the wiki.
 *
 * Author:   Joachim Wuttke <j.wuttke@fz-juelich.de> 2004-2013
 * 
 * Licence:  see ../COPYING (FreeBSD)
 * 
 * Homepage: apps.jcns.fz-juelich.de/lmfit
 */

#define EXPORT __declspec(dllexport)
 
#include "lmcurve.h"
#include <stdio.h>
#include <math.h>
/* model function: a parabola */

double f( double t, const double *p )
{
    //printf("test %f\n",(p[0] + p[1]*t + p[2]*t*t));
	//printf("test abs %f\n",fabs(p[0] + p[1]*t + p[2]*t*t));
	return 	fabs(p[0]-p[1]*1.0*exp(-t/p[2]));

}

EXPORT void __stdcall T1map(double *t1,
                                 double *y1,
                                 double *col) 

{
    int n = 3; /* number of parameters in model function f */
    double par[3] = { 100, 200, 3000 }; /* really bad starting value */
    
    /* data points: a slightly distorted standard parabola */
   
    int m = 11;
    int i;

    double t[11] = { 120.2, 220.3, 370.1, 1130.5, 1168, 1233, 2115, 2125, 2145, 3078, 4035 };
    double y[11] = { 114.2, 87., 56., 75., 80., 89., 137., 132.4, 128., 151, 168.5};

    lm_control_struct control = lm_control_double;
	//lm_control_struct control = {1.e-12, 1.e-12, 1.e-12, 1.e-12, 100., 100, 0, 0 };
    lm_status_struct status;
    control.verbosity = 9;

    printf( "Fitting ...\n" );
    /* now the call to lmfit */
    lmcurve( n, par, m, t1, y1, f, &control, &status );
        
    printf( "Results:\n" );
    printf( "status after %d function evaluations:\n  %s\n",
            status.nfev, lm_infmsg[status.outcome] );

    printf("obtained parameters:\n");
    for ( i = 0; i < n; ++i)
        printf("  par[%i] = %12g\n", i, par[i]);
    printf("obtained norm:\n  %12g\n", status.fnorm );
    
    printf("fitting data as follows:\n");
    for ( i = 0; i < m; ++i)
        printf( "  t[%2d]=%4g y=%6g fit=%10g residue=%12g\n",
                i, t[i], y[i], f(t[i],par), y[i] - f(t[i],par) );

    col[0] =par[0];
	col[1] =par[1];
	col[2] =par[2];
	//return 0;
}
