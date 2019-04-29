/*!
  * \file
  * \brief Include this into the main file to have access to the MHD library and code.
  */
#pragma once

// IMPORTANT: Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on Armadillo objects
#define VIENNACL_WITH_ARMADILLO

// std includes
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
// external libraries
#include <armadillo>
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/lu.hpp"
#include "viennacl/linalg/sum.hpp"
#include "viennacl/linalg/maxmin.hpp"
#include "viennacl/tools/random.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/forwards.h"

typedef viennacl::vector<double> vcl_vec; /*!< GPU arma::vector with ViennaCL */
typedef viennacl::compressed_matrix<double> vcl_sp_mat; /*!< GPU sparse matrix with ViennaCL */
typedef viennacl::matrix<double> vcl_mat; /*!< GPU dense matrix with ViennaCL */

// local includes
#include "params.h"
#include "types.h"
#include "fluxl.h"
#include "operators.h"

namespace comfi
{

/*!
 * Utility functions. This namespace contains non-mathematical functions to assist with IO, monitoring, logging and error checking. Also included are RHS and LHS functions.
 */
namespace util
{
/*!
 * \brief gettimestr Get current time.
 * \return const std::string in YYYY-MM-DD-HH-MM format.
 */
std::string gettimestr();

/*!
 * \brief sendtolog Send message to log file.
 * \param message Message std::string to show in log file.
 * \param filename The log filename make sure this is always the same.
 */
void sendtolog(const std::string message, const std::string filename);

/*!
 * \brief fillInitialCondition fills a mhdsim::types::BgData with the initial condition
 * \param ctx The mhdsim::types::BgData context to use to fill.
 * \return a result arma::vector x0 that is the initial condition.
 */
arma::vec fillInitialCondition(const comfi::types::BgData &ctx);

/*!
 * \brief Calculate the initial condition based on reconnection conditions.
 * \param bg Background data, this will be changed in the function.
 * \param op operators
 * \return initial condition vector
 */
std::tuple<arma::vec, const comfi::types::BgData> calcReconnectionIC(const comfi::types::Operators &op);

/*!
 * \brief Calculate the initial condition based on Soler et al 2013.
 * \param bg Background data, this will be changed in the function.
 * \param op operators
 * \return initial condition vector
 */
std::tuple<arma::vec, const comfi::types::BgData> calcSolerIC(const comfi::types::Operators &op);

/*!
 * \brief Calculate the initial condition based on Sod's Shock Tube.
 * \param bg Background data, this will be changed in the function.
 * \param op operators
 * \return initial condition vector
 */
std::tuple<arma::vec, const comfi::types::BgData> calcShockTubeIC(const comfi::types::Operators &op);

/*!
 * \brief Calculate the initial condition based on scalar magnetic potential solution.
 * \param bg Background data, this will be changed in the function.
 * \param op operators
 * \return initial condition vector
 * Solves $[ \nabla^2 \Phi $] and fills initial condition. Density is calculated hydrostatically.
 */
std::tuple<arma::vec, const comfi::types::BgData> calcInitialCondition(const types::Operators &op);

/*!
 * \brief Check in temperatures are negative (sane is all positive)
 * \param xn result arma::vector
 * \param op operators
 * \return true if sane, false if insane (negatives found)
 */
bool sanityCheck(vcl_vec &xn, const comfi::types::Operators &op);

// Misc
/*!
 * \brief Creates a sparse matrix after async parallel construction of i,j,val components.
 * \param Avi Vector with i indeces
 * \param Avj Vector with j indeces
 * \param Avv Vector with values
 * \param num_of_rows Number of rows in sparse matrix
 * \param num_of_cols Number of columns in sparse matrix
 * \return cpu sparse matrix
 */
arma::sp_mat syncSpMat(const arma::umat Avi, const arma::umat Avj, const arma::mat Avv, const uint num_of_rows =num_of_elem, const uint num_of_cols =num_of_elem);

/*!
 * \brief getmaxV get fast mode speed + local speed OR resistivity speed whichever is faster
 * \param x0 result arma::vector
 * \param op operators
 * \return speed in normalized units
 */
double getmaxV(const vcl_vec &x0, const comfi::types::Operators &op);

/*!
 * \brief saveSolution Save solution into seperate binary files for each unknown.
 * \param x0 result arma::vector
 * \param timestep The time step of the solution
 * \param op operators
 * \return True if no errors.
 */
bool saveSolution(const vcl_vec &x0, const int &timestep, comfi::types::Operators &op);

/*!
 * \brief Calculate the magnetic field energy density sum of all grid cells.
 * \param x0 result vcl_vec
 * \param op operators
 * \return sum of all magnetic field energy densities
 */
double getsumBE(const vcl_vec &x0, const comfi::types::Operators &op);

/*!
 * \brief Calculate the kinetic energy density sum of all grid cells.
 * \param x0 result vcl_vec
 * \param op operators
 * \return sum of all kinetic energy densities
 */
double getsumKE(const vcl_vec &x0, const comfi::types::Operators &op);

/*!
 * \brief Calculate the thermal energy density sum of all grid cells.
 * \param x0 result vcl_vec
 * \param op operators
 * \return sum of all thermal energy densities
 */
double getsumUE(const vcl_vec &x0, const comfi::types::Operators &op);

/*!
 * \brief Calculate the magnetic field energy density sum of all grid cells.
 * \param x0 result arma::vector
 * \return Sum of all magnetic field energy densities.
 */
double getsumBE(const arma::vec &x0);

/*!
 * \brief Calculate the sum of the kinetic energy of all grid cells.
 * \param x0 result arma::vector
 * \return Sum of all kinetic energy densities (all species).
 */
double getsumKE(const arma::vec &x0);

/*!
 * \brief Calculate the sum of the thermal energy density of all grid cells.
 * \param x0 result arma::vector
 * \return Sum of all thermal energy densities (all species).
 */
double getsumUE(const arma::vec &x0);

/*!
 * \brief Write a binary file of a field arma::vector. Seperate binary files for each direction.
 * \param x0 result arma::vector
 * \param name Name of field, will be used as filename prefix.
 * \param timestep Time step to save as
 * \return True if no error.
 */
bool saveField(const arma::vec &x0, const std::string name, const int timestep);

/*!
 * \brief saveScalar Write a binary file of a scalar arma::vector.
 * \param x0 result arma::vector
 * \param name Name of scalar, will be used as filename prefix.
 * \param timestep Time step to save as
 * \return True if no error.
 */
bool saveScalar(const arma::vec &x0, const std::string name, const int timestep);

/*!
 * \brief saveField Write a binary file of a field arma::vector. Seperate binary files for each direction.
 * \param x0 result arma::vector
 * \param name Name of field, will be used as filename prefix.
 * \param timestep Time step to save as
 * \return True if no error.
 */
bool saveField(const vcl_vec &x0, const std::string name, const int timestep);

/*!
 * \brief saveScalar Write a binary file of a scalar arma::vector.
 * \param x0 result arma::vector
 * \param name Name of scalar, will be used as filename prefix.
 * \param timestep Time step to save as
 * \return True if no error.
 */
bool saveScalar(const vcl_vec &x0, const std::string name, const int timestep);

} // namespace util

namespace routines
{

vcl_mat Re_MUSCL(const vcl_mat &xn, const double t, comfi::types::Context &ctx);

/*!
 * \brief Flux limiters
 * \param r ratio of gradients
 * \return Limiter function values
 */
vcl_mat fluxl(const vcl_mat &r);

void topbc_driver(vcl_vec &Lxn, vcl_vec &Rxn, const double t, const comfi::types::Operators &op);

/*!
 * \brief Get neutral sound speed
 * \param xn matrix of results
 * \param ctx Simulation context
 * \return vector of neutral sound speed
 */
vcl_vec sound_speed_neutral(const vcl_mat &xn, const comfi::types::Context &ctx);

/*!
 * \brief fast_speed_x Get vertical fast mode speed
 * \param xn result matrix
 * \param ctx Simulation context.
 * \return Column vector of vertical fast mode speed of ions.
 */
vcl_vec fast_speed_z(const vcl_mat &xn, const comfi::types::Context &ctx);

/*!
 * \brief fast_speed_x Get horizontal fast mode speed
 * \param xn Result matrix.
 * \param ctx	Simulation context.
 * \return Column vector of horizontal fast mode speeds.
 */
vcl_vec fast_speed_x(const vcl_mat &xn, const comfi::types::Context &ctx);

/*!
 * \brief Get largest vertical speed
 * \param xn result
 * \param op operators
 * \return scalar of vertical speeds
 */
vcl_vec cz_max(const vcl_vec &xn, const comfi::types::Operators &op);

/*!
 * \brief Get largest horizontal speed
 * \param xn result
 * \param op operators
 * \return scalar of horizontal speeds
 */
vcl_vec cx_max(const vcl_vec &xn, const comfi::types::Operators &op);

/*!
 * \brief getCzMax Get vertical fast mode speed
 * \param xn result arma::vector
 * \param op operators
 * \return scalar arma::vector of vertical fast mode speeds
 */
vcl_vec fast_speed_z(const vcl_vec &xn, const comfi::types::Operators &op);

/*!
 * \brief polyval compute's polynomial using Herner's scheme
 * \param p polynomial coefficient vector of size (polynomial degree + 1)
 * \param x vector to be computed
 * \return result of the polynomial
 */
vcl_vec polyval(const arma::vec &p, const vcl_vec &x);

/*!
 * \brief cross_prod Cross product of two fields
 * \param f1 Field arma::vector to cross product with f1 x f2
 * \param f2 Field arma::vector to cross product by f1 x f2
 * \param op operators
 * \return field arma::vector that is the result = f1 x f2
 */
vcl_vec cross_prod(const vcl_vec &f1, const vcl_vec &f2, const comfi::types::Operators &op);

/*!
 * \brief dot_prod Dot product two field arma::vectors
 * \param f1 Field arma::vector to dot product f1 . f2
 * \param f2 Field arma::vector to dot product f1 . f2
 * \param op operators
 * \return scalar arma::vector that is the result of = f1 . f2
 */
vcl_vec dot_prod(const vcl_vec &f1, const vcl_vec &f2, const comfi::types::Operators &op);

/*!
 * \brief computeRHS_Euler Compute right hand side using Eulerian time stepping.
 * \param xn Current time step result arma::vector
 * \param dt Change in time
 * \param t Time elapsed
 * \param op operators
 * \param bg Background data
 * \return Right hand side result arma::vector
 */
vcl_vec computeRHS_Euler(const vcl_vec &xn, const double dt, const double t, const comfi::types::Operators &op, const comfi::types::BgData &bg);

/*!
 * \brief computeRHS_RK4 Compute right hand side using Runge-Kutta 4 time stepping.
 * \param xn Current time step result arma::vector
 * \param dt Change in time
 * \param t Time elapsed
 * \param op operators
 * \param bg Background data
 * \return  Right hand side result arma::vector
 */
vcl_vec computeRHS_RK4(const vcl_vec &xn, const double dt, const double t, const comfi::types::Operators &op, const comfi::types::BgData &bg);

/*!
 * \brief computeRHS_BDF2 Compute right hand side using semi-implicit BDF2 time stepping.
 * \param xn current time step result arma::vector
 * \param xn1 previous time step result arma::vector
 * \param Ri implicit part
 * \param alpha BDF2 parameter
 * \param beta BDF2 parameter
 * \param dt Change in time
 * \param t Time elapsed
 * \param op Operators
 * \param bg Background data
 * \return Right hand side result arma::vector
 */
vcl_vec computeRHS_BDF2(const vcl_vec &xn, const vcl_vec &xn1, const vcl_sp_mat &Ri, const double alpha, const double beta, const double dt, const double t, comfi::types::Operators &op, const comfi::types::BgData &bg);

/*!
 * \brief Calculate implicit part (not including the identity sparse matrix)
 * \param xn_vcl result vec_vcl
 * \param op operators
 * \return Sparse matrix of implicit terms.
 */
arma::sp_mat computeRi(const vcl_vec &xn_vcl, const comfi::types::Operators &op);

/*!
 * \brief Soler dirichlet boundary conditions
 * \param Lxn Left state vcl_vector
 * \param Rxn Right state vcl_vector
 * \param op Operators
 */
void bottombc_soler(vcl_vec &Lxn, vcl_vec &Rxn, const comfi::types::Operators &op);

/*!
 * \brief Soler dirichlet boundary conditions
 * \param Lxn Left state vcl_vector
 * \param Rxn Right state vcl_vector
 * \param op Operators
 */
void topbc_soler(vcl_vec &Lxn, vcl_vec &Rxn, const comfi::types::Operators &op);

/*!
 * \brief Shock Tube dirichlet boundary conditions
 * \param Lxn Left state vcl_vector
 * \param Rxn Right state vcl_vector
 * \param op Operators
 */
void topbc_shock_tube(vcl_vec &Lxn, vcl_vec &Rxn, const comfi::types::Operators &op);

/*!
 * \brief Shock Tube dirichlet boundary conditions
 * \param Lxn Left state vcl_vector
 * \param Rxn Right state vcl_vector
 * \param op Operators
 */
void bottombc_shock_tube(vcl_vec &Lxn, vcl_vec &Rxn, const comfi::types::Operators &op);

/*!
 * \brief bottomBC Fill bottom boundary with photosphere activty.
 * \param Lxn Left state arma::vector from MUSCL scheme
 * \param Rxn Right state arma::vector from MUSCL scheme
 * \param t Time elapsed
 * \param op Operators
 * \param bg Background data
 */
void bottomBC(vcl_vec &Lxn, vcl_vec &Rxn, const double t, const comfi::types::Operators &op, const comfi::types::BgData &bg);

/*!
 * \brief Fill bottom boundary with square perp velocity.
 * \param Lxn Left state arma::vector from MUSCL scheme
 * \param Rxn Right state arma::vector from MUSCL scheme
 * \param t Time elapsed
 * \param op Operators
 * \param bg Background data
 */
void bottomBCsquare(vcl_vec &Lxn, vcl_vec &Rxn, const double t, const comfi::types::Operators &op, const comfi::types::BgData &bg);

/*!
 * \brief Fx Calculate horizontal flux value at cell edge.
 * \param xn Cell edge result arma::vector
 * \param Npij Cell center ion density scalar arma::vector to divide Temp flux by
 * \param op Operators
 * \return result arma::vector Fx at cell edge
 */
vcl_vec Fx(const vcl_vec &xn,const vcl_vec &Npij, const comfi::types::Operators &op);

/*!
 * \brief Fz Calculate vertical flux value at cell edge.
 * \param xn Cell edge result arma::vector
 * \param Npij Cell center ion density scalar arma::vector to divide Temp flux by
 * \param op Operators
 * \return result arma::vector Fz at cell edge
 */
vcl_vec Fz(const vcl_vec &xn,const vcl_vec &Npij, const comfi::types::Operators &op);

vcl_vec Re_MUSCL(const vcl_vec &xn, const double t, const comfi::types::Operators &op, const comfi::types::BgData &bg);

// inlines
inline const vcl_vec grad_tvd(const vcl_vec &s_iph,
            const vcl_vec &s_imh,
            const vcl_vec &s_jph,
            const vcl_vec &s_jmh,
            const comfi::types::Operators &op)
{
  const vcl_vec xpart = (s_iph-s_imh)/dx;
  const vcl_vec zpart = (s_jph-s_jmh)/dz;
  return viennacl::linalg::prod(op.s2xf,xpart) + viennacl::linalg::prod(op.s2zf,zpart);
}

inline const vcl_vec div_tvd(const vcl_vec &s_iph, const vcl_vec &s_imh, const vcl_vec &s_jph, const vcl_vec &s_jmh)
{
  const vcl_vec xpart = (s_iph-s_imh)/dx;
  const vcl_vec zpart = (s_jph-s_jmh)/dz;
  return xpart+zpart;
}

inline const vcl_vec curl_tvd(const vcl_vec &f_iph, const vcl_vec &f_imh, const vcl_vec &f_jph, const vcl_vec &f_jmh, const comfi::types::Operators &op)
{
  using namespace viennacl::linalg;
  const vcl_vec xpart = (prod(op.fdotp, f_jph) - prod(op.fdotp, f_jmh))/dz;
  const vcl_vec zpart = (prod(op.fdotp, f_imh) - prod(op.fdotp, f_iph))/dx;
  const vcl_vec ppart = (prod(op.fdotz, f_iph) - prod(op.fdotz, f_imh))/dx - (prod(op.fdotx, f_jph) - prod(op.fdotx, f_jmh))/dz;
  return prod(op.s2xf, xpart) + prod(op.s2zf, zpart) + prod(op.s2pf, ppart);
}


} // namespace routines

/*!
 * Solar chromosphere math functions.
 */
namespace sol
{
inline double nu_nn(const double &nn, const double &T)
{
  const double sigma_nn = 7.73e-19; //m-2
  const double m_nn = 1.007825*arma::datum::m_u;
  const double nn_coeff = sigma_nn * std::sqrt(16.0*arma::datum::k/(arma::datum::pi*m_nn));
  return (nn_coeff * nn * n_0 * std::sqrt(std::abs(T*T_0)) * t_0); // ion-neutral collision rate
}

inline arma::vec nu_nn(const arma::vec &nn, const arma::vec &T)
{
  const double sigma_nn = 7.73e-19; //m-2
  const double m_nn = 1.007825*arma::datum::m_u;
  const double nn_coeff = sigma_nn * std::sqrt(16.0*arma::datum::k/(arma::datum::pi*m_nn));
  return (nn_coeff * n_0 * t_0 * nn % arma::sqrt(arma::abs(T*T_0)) ); // ion-neutral collision rate
}

inline vcl_vec nu_nn(const vcl_vec &nn, const vcl_vec &T)
{
  const double sigma_nn = 7.73e-19; //m-2
  const double m_nn = 1.007825*arma::datum::m_u;
  const double nn_coeff = sigma_nn * std::sqrt(16.0*arma::datum::k/(arma::datum::pi*m_nn));
  return nn_coeff * n_0 * t_0  * viennacl::linalg::element_prod(nn, viennacl::linalg::element_sqrt(T*T_0)); // ion-neutral collision rate
}

inline double nu_in(const double &nn, const double &T)
{
  const double sigma_in = 1.16e-18; //m-2
  const double m_in = 1.007276466879*1.007825/(1.007276466879+1.007825);
  const double in_coeff = sigma_in * std::sqrt(8.0*arma::datum::k/(arma::datum::pi*m_in));
  return (in_coeff * nn * n_0 * std::sqrt(std::abs(T*T_0)) * t_0); // ion-neutral collision rate
}

inline arma::vec nu_in(const arma::vec &nn, const arma::vec &T)
{
  const double sigma_in = 1.16e-18; //m-2
  const double m_in = 1.007276466879*1.007825/(1.007276466879+1.007825);
  const double in_coeff = sigma_in * std::sqrt(8.0*arma::datum::k/(arma::datum::pi*m_in));
  return (in_coeff * n_0 * t_0 * nn % arma::sqrt(arma::abs(T*T_0)) ); // ion-neutral collision rate
}

inline vcl_vec nu_in(const vcl_vec &nn, const vcl_vec &T)
{
  const double sigma_in = 1.16e-18; //m-2
  const double m_in = 1.007276466879*1.007825/(1.007276466879+1.007825);
  const double in_coeff = sigma_in * std::sqrt(8.0*arma::datum::k/(arma::datum::pi*m_in));
  return in_coeff * n_0 * t_0  * viennacl::linalg::element_prod(nn, viennacl::linalg::element_sqrt(T*T_0)); // ion-neutral collision rate
}

inline double nu_en(const double &nn, const double &T)
{
  const double sigma_en = 1.0e-19; //m-2
  const double m_en = arma::datum::m_e*1.007825*arma::datum::m_u/(arma::datum::m_e+1.007825*arma::datum::m_u);
  const double en_coeff = sigma_en * std::sqrt(8.0*arma::datum::k/(arma::datum::pi*m_en));
  return (en_coeff * nn * n_0 * sqrt(std::abs(T*T_0)) * t_0); // electron-neutral collision rate
}

inline vcl_vec nu_en(const vcl_vec &nn, const vcl_vec &T)
{
  const double sigma_en = 1.0e-19; //m-2
  const double m_en = arma::datum::m_e*1.007825*arma::datum::m_u/(arma::datum::m_e+1.007825*arma::datum::m_u);
  const double en_coeff = sigma_en * std::sqrt(8.0*arma::datum::k/(arma::datum::pi*m_en));
  return en_coeff * n_0 * t_0 * viennacl::linalg::element_prod(nn, viennacl::linalg::element_sqrt(T*T_0)); // electron-neutral collision rate
}

inline double nu_ei(const double &ne, const double &T)
{
  const double coloumb_logarithm = 10.0;
  const double r = arma::datum::ec*arma::datum::ec/(4.0*arma::datum::pi*arma::datum::eps_0*arma::datum::k);
  const double sigma_ei = coloumb_logarithm*arma::datum::pi*r*r;
  const double coeff_ei = (4.0/3.0) * sigma_ei * std::sqrt(2.0*arma::datum::k/(arma::datum::pi*arma::datum::m_e));
  return (coeff_ei * ne * n_0 * std::pow(std::abs(T*T_0),-1.5) * t_0); // electron-ion collision rate
}

inline vcl_vec nu_ei(const vcl_vec &ne, const vcl_vec &Te)
{
  const double coloumb_logarithm = 10.0;
  const double r = arma::datum::ec*arma::datum::ec/(4.0*arma::datum::pi*arma::datum::eps_0*arma::datum::k);
  const double sigma_ei = coloumb_logarithm*arma::datum::pi*r*r;
  const double coeff_ei = (4.0/3.0) * sigma_ei * std::sqrt(2.0*arma::datum::k/(arma::datum::pi*arma::datum::m_e));
  return coeff_ei * n_0 * t_0* viennacl::linalg::element_prod(ne,viennacl::linalg::element_exp(-1.5*viennacl::linalg::element_log(T_0*Te))); // electron-ion collision rate
}

inline vcl_vec nu_ii(const vcl_vec &Np, const vcl_vec &Tp)
{
  const double coloumb_logarithm = 10.0;
  const double r = arma::datum::ec*arma::datum::ec/(4.0*arma::datum::pi*arma::datum::eps_0*arma::datum::k); //without T
  const double sigma_ii = coloumb_logarithm*arma::datum::pi*r*r;
  const double coeff_ii = (4.0/3.0) * sigma_ii * std::sqrt(2.0*arma::datum::k/(arma::datum::pi*arma::datum::m_p));
  return t_0 * n_0 * coeff_ii * viennacl::linalg::element_prod(Np, viennacl::linalg::element_exp(-1.5*viennacl::linalg::element_log(T_0*Tp))); // electron-ion collision rate
}

inline double nu_ii(const double &Np, const double &Tp)
{
  const double coloumb_logarithm = 10.0;
  const double r = arma::datum::ec*arma::datum::ec/(4.0*arma::datum::pi*arma::datum::eps_0*arma::datum::k); //without T
  const double sigma_ii = coloumb_logarithm*arma::datum::pi*r*r;
  const double coeff_ii = (4.0/3.0) * sigma_ii * std::sqrt(2.0*arma::datum::k/(arma::datum::pi*arma::datum::m_p));
  return coeff_ii * t_0 * Np * n_0 * std::exp(-1.5*std::log(T_0*Tp)); // electron-ion collision rate
}

inline vcl_vec resistivity(const vcl_vec &Np, const vcl_vec &Nn, const vcl_vec &Tp, const vcl_vec &Tn)
{
  return (m_e/(q*q))*viennacl::linalg::element_div((nu_ei(Np,Tp)+nu_en(Nn,0.5*(Tp+Tn))),Np);
}
inline double resistivity(const double &Np, const double &Nn, const double &Tp, const double &Tn)
{
  return (m_e/(q*q)) * (nu_ei(Np,Tp)+nu_en(Nn,0.5*(Tp+Tn)))/Np;
}

inline double thermalvel(const double &T)
{
  return std::sqrt(2.0*arma::datum::k/arma::datum::m_p) * std::sqrt(T*T_0) / V_0;
}

inline vcl_vec thermalvel(const vcl_vec &T)
{
  using namespace viennacl::linalg;
  return std::sqrt(2.0*arma::datum::k/arma::datum::m_p) * element_sqrt(T*T_0) / V_0;
}

inline double kappa_n(const double &Tp, const double &Tn, const double &Np, const double &Nn)
{
  return Nn * thermalvel(Tn) * thermalvel(Tn) / nu_nn(Nn, Tn);
}

inline vcl_vec kappa_n(const vcl_vec &Tp, const vcl_vec &Tn, const vcl_vec &Np, const vcl_vec &Nn)
{
  using namespace viennacl::linalg;
  const vcl_vec v2 = element_prod(thermalvel(Tn), thermalvel(Tn));
  const vcl_vec nv2 = element_prod(Nn, v2);
  const vcl_vec nunn = nu_nn(Nn, Tn);
  return element_div(nv2, nunn);
}

inline double kappa_p(const double &Tp, const double &Tn, const double &Np, const double &Nn)
{
  return Np * thermalvel(Tp) * thermalvel(Tp) / (nu_in(Nn, 0.5*(Tp+Tn)) + nu_ii(Np, Tp));
}

inline vcl_vec kappa_p(const vcl_vec &Tp, const vcl_vec &Tn, const vcl_vec &Np, const vcl_vec &Nn)
{
  using namespace viennacl::linalg;
  const vcl_vec v2 = element_prod(thermalvel(Tp), thermalvel(Tp));
  const vcl_vec nv2 = element_prod(Np, v2);
  const vcl_vec nuin = nu_in(Nn, 0.5*(Tp+Tn));
  const vcl_vec nuii = nu_ii(Np, Tp);
  return element_div(nv2, nuin+nuii);
}

/*!
 * \brief Calculate recombination coefficient for ions to recombine to neutrals.
 * \param Tp Ion temperature
 * \return Vector of normalized volumetric recombination rate.
 */
inline vcl_vec recomb_coeff(const vcl_vec &T)
{
  using namespace viennacl::linalg;
  static const vcl_vec one = viennacl::scalar_vector<double>(T.size(), 1.0);
  const vcl_vec beta = 0.6 * 13.6 * 1.6021766208e-19 * element_div(one, arma::datum::k*T_0*T);
  vcl_vec coeff = 0.4288*one + 0.5*element_log(beta) + 0.4698*element_pow(beta, -one_third*one);
  coeff = element_prod(5.20e-20*element_sqrt(beta), coeff);
  return t_0 * n_0 * coeff;
}

inline double recomb_coeff(const double &T)
{
  const double beta = 0.6 * 13.6 * 1.6021766208e-19 / (arma::datum::k*T_0*T);
  return t_0 * n_0 * 5.20e-20 * std::sqrt(beta) * (0.4288+std::log(beta)+0.4698*std::pow(beta, -one_third));
}

/*!
 * \brief Calculate normalized ionization coefficient for neutrals to ionize due to collisions (Draine p 134)
 * \param Tn Neutral temperature
 * \return Vector of normalized volumetric ionization rate
 */
inline vcl_vec ionization_coeff(const vcl_vec &T)
{
  using namespace viennacl::linalg;
  static const vcl_vec one = viennacl::scalar_vector<double>(T.size(), 1.0);
  const vcl_vec beta = 0.6 * 13.6 * 1.6021766208e-19 * element_div(one, arma::datum::k*T_0*T);
  const vcl_vec coeff = element_div(2.34e-14*element_exp(-beta), element_sqrt(beta));
  return t_0 * n_0 * coeff;

}

inline double ionization_coeff(const double &T)
{
  const double beta = 0.6 * 13.6 * 1.6021766208e-19 / (arma::datum::k*T_0*T);
  return t_0 * n_0 * 2.34e-14 * std::exp(-beta) / std::sqrt(beta);
}
} // namespace sol

} // namespace mhdsim


/*!
 * \brief Find scalar vector index.
 * \param i Horizontal index.
 * \param j Vertical index
 * \return Scalar vector index.
 */
inline arma::uword inds(const arma::uword &i, const arma::uword &j, comfi::types::Context &ctx)
{
  return i+j*ctx.nx();
}
