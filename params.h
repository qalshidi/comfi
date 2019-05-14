/*!
  * \file
  * \brief Parameters are editable here (such as grid size)
  */
#pragma once

#include <cmath>
#include <armadillo>
#include "comfi.h"

const unsigned short num_of_eq  = 14;
const unsigned short DOF        = 3;
/// m-3 density at z=0
const double  n_0 = 1;
/// in T
const double  B_0 = 1;
/// num of grid points in horizontal direction
const uint nx = 1;
/// num of grid points in vertical direction
const uint nz = 1001;

const unsigned int num_of_grid = nx*nz;
const unsigned int num_of_elem = num_of_eq*nx*nz;
const unsigned int num_of_species = 2;

// Normalization and constants (SI units)
/// ratio of specific heats
const double gammamono = 5.0/3.0;
/// π
const double  pi                = arma::datum::pi;
/// mass of proton in kg
const double  m_i               = arma::datum::m_p;
/// permeability of free space
const double  mu_0              = arma::datum::mu_0;
/// boltzmann constant in SI
const double  k_b               = arma::datum::k;
/// elementary charge in C
const double  e_                = arma::datum::ec;

// Derived normalization and constants
/// mass of electron in kg
const double m_e               = arma::datum::m_e/m_i;
const double p_0               = B_0*B_0/mu_0;
const double T_0               = p_0/(n_0*k_b);
/// in m/s
const double V_0               = B_0/std::sqrt(mu_0*n_0*m_i);
//const double V_0 = std::sqrt(gammamono*p_0/n_0);

/// Length scale in m
const double l_0 = 1.0;
/// Chromospheric height in the bottom boundary in m
const double height_start = 0.0;
/// Chromospheric height in the bottom boundary in m
const double height_end = 1.0;
/// Height of box in m
const double height = l_0;
/// Width of box in m
const double width = l_0;

/// in s
const double t_0               = l_0/V_0;
/// in m/s2 then normalized
const double g                 = 0.27395e3*t_0/V_0;
const double q_0               = p_0*V_0;
const double e_0               = B_0/(mu_0*n_0*V_0*l_0);
const double q                 = e_/e_0;
const double kappa_0           = q_0*l_0/T_0;

const double dx = (width/nx)/l_0; const double dz = (height/nz)/l_0;
const double dx2 = dx*dx; const double dz2 = dz*dz;
//const double ds = dx*(dx<=dz)+dz*(dz<dx);
const double ds = dz;

/// isothermal temp IC
const double T0 = 6580.0/T_0;
const double Np0 = 8.40000000e+19/n_0;
const double Nn0 = 1.19000000e+23/n_0;
const double collisionrate = 0.1*V_0/l_0;

const double  five_thirds = 5.0/3.0;   const double one_third = 1.0/3.0; const double two_thirds = 2.0/3.0;
const double  five_sixths = 5.0/6.0;

// Solution index
/// Result vector top level index
const arma::uword Bx    = 0;
/// Result vector top level index
const arma::uword Bz    = 1;
/// Result vector top level index
const arma::uword Bp    = 2;
/// Result vector top level index
const arma::uword n_n   = 3;
/// Result vector top level index
const arma::uword Ux    = 4;
/// Result vector top level index
const arma::uword Uz    = 5;
/// Result vector top level index
const arma::uword Up    = 6;
/// Result vector top level index
const arma::uword n_p   = 7;
/// Result vector top level index
const arma::uword Vx    = 8;
/// Result vector top level index
const arma::uword Vz    = 9;
 /// Result vector top level index
const arma::uword Vp    = 10;
/// Result vector top level index
const arma::uword E_n   = 11;
 /// Result vector top level index
const arma::uword E_p   = 12;
/// Result vector top level index
const arma::uword GLM   = 13;
/// Field vector top level index
const arma::uword _x    = 0;
 /// Field vector top level index
const arma::uword _z    = 1;
/// Field vector top level index
const arma::uword _p    = 2;

/*!
 * \brief Find scalar vector index.
 * \param i Horizontal index.
 * \param j Vertical index
 * \return Scalar vector index.
 */
inline arma::uword inds(const arma::uword i, const arma::uword j)
{
  return i+j*nx;
}
/*!
 * \brief Find result vector index. Add a solution index (e.g: ind(0, 0)+Ux) to find value.
 * \param i Horizontal index.
 * \param j Vertical index.
 * \return Result vector index.
 */
inline int ind(const arma::uword &var, const arma::uword &i, const arma::uword &j)
{
  return var*num_of_grid+inds(i, j);
}

/*!
 * \brief Find result vector index. Add a solution index (e.g: ind(0, 0)+Ux) to find value.
 * \param i Horizontal index.
 * \param j Vertical index.
 * \return Result vector index.
 */
inline int ind(const uint i, const uint j)
{
  return i*num_of_eq+j*nx*num_of_eq;
}

/*!
 * \brief Find result vector index. Add a solution index (e.g: ind(0, 0)+Ux) to find value.
 * \param i horizontal index
 * \param j vertical index
 * \return Result vector index.
 */
inline arma::urowvec ind(const arma::urowvec& i, const arma::urowvec& j)
{
  return i*num_of_eq + j*nx*num_of_eq;
}

/*!
 * \brief Find result vector index. Add a solution index (e.g: ind(0, 0)+Ux) to find value.
 * \param i horizontal index
 * \param j vertical index
 * \return Result vector index.
 */
inline const arma::urowvec ind(const arma::urowvec& i, const arma::urowvec& j, const uint& unknown)
{
  return unknown*arma::ones<arma::urowvec>(i.n_cols) + i*num_of_eq + j*nx*num_of_eq;
}

/*!
 * \brief Find field vector index. Add a component index (e.g: indf(0, 0)+_x) to find value.
 * \param i Horizontal index.
 * \param j Vertical index.
 * \return Field vector index.
 */
inline int indf(const uint i, const uint j)
{
  return i*DOF+j*nx*DOF;
}

