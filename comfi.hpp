/*!
  * \file
  * \brief Include this into the main file to have access to the MHD library and code.
  */
#pragma once

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


/*!
 * Everything in CoMFi will be in this namespace for safety. This is the top level namespace.
 */
namespace comfi
{

/*!
* All the classes, enums and types can be found here.
*/
namespace types {

/*!
 * \struct Settings
 * \brief The settings for the simulation (used in main loop) are here.
 */
struct Settings {
  /// Maximum number of time steps before simulation end.
  int max_time_steps = -1;
  /// Maximum time before simulation end in normalized units. \f$(t_0)\f$
  double max_time = -1.0;
  /// gmres tolerance
  double tolerance = 1.e-6;
  /// Save solution every X time steps
  int save_dn = -1;
  /// Save solution every X time steps
  double save_dt = -1;
  /// If this run restarts from a previous run
  bool restart = false;
  /// Other flags for runtime.
  uint flags = 0;
};

/*!
 * \brief The Boundary Condition enum.
 */
enum BoundaryCondition {PERIODIC, /*!< Periodic boundary condition (i+1 at right boundary is i=0) */
                        DIRICHLET, /*!< Dirichlet boundary conditions. Inserting data up to user. */
                        NEUMANN, /*!< Neumann boundary conditions. (i+1 at right boundary = i) */
                        MIRROR, /*!< Mirror boundary conditions. (i+1 at right boundary = i or negative that value in case of x component) */
                        DIMENSIONLESS /*!< This boundary is not in a free dimension. */
                       };

/*!
 * \brief The simulation context. Everything needed about the simulation is in here.
 */
class Context {
  /// Current \f$\Delta t\f$ in normalized units of \ref t_0 .
  double m_dt = 0.0;
  /// Current \f$t\f$ in normalized units of \ref t_0 .
  double m_time_elapsed = 0.0;
  /// Current step \f$n\f$ .
  uint m_time_step = 0;

public:
  // Simulation domain
  /// Number of grid points in x-direction (horizontal).
  const arma::uword nx;
  /// Number of grid points in z-direction (vertical).
  const arma::uword nz;
  /// Upper boundary condition in the z-direction.
  const BoundaryCondition bc_up;
  /// Lower boundary condition in the z-direction.
  const BoundaryCondition bc_down;
  /// Right boundary condition in the x-direction.
  const BoundaryCondition bc_right;
  /// Left boundary condition in the x-direction.
  const BoundaryCondition bc_left;

  /// Settings passed on from main.
  const Settings settings;

  // Constants
  /// ratio of specific heats
  const double gammamono = 5.0/3.0;
  /// π
  const double  pi = arma::datum::pi;
  /// mass of proton in kg
  const double m_i = arma::datum::m_p;
  /// permeability of free space
  const double  mu_0 = arma::datum::mu_0;
  /// boltzmann constant in SI
  const double  k_b = arma::datum::k;
  /// elementary charge in C
  const double  e_ = arma::datum::ec;
  // Normalization constants
  /// Length normalization constant in meters.
  const double l_0 = 2.1e6;
  /// \f$m^{-3}\f$ density normalization constant.
  const double n_0 = 1.e20;
  /// Magnetic field normalization constant in Teslas.
  const double B_0 = 0.1;
  /// Width (x-direction) in normalized units.
  const double width = l_0;
  /// Height (z-direction) in normalized units.
  const double height = l_0;
  // Derived constants
  /// \f$\Delta x\f$ in normalized units.
  const double dx = (width/nx)/l_0;
  /// \f$\Delta z\f$ in normalized units.
  const double dz = (height/nz)/l_0;
  /// \f$\Delta s\f$ (smallest grid length) in normalized units.
  const double ds = (dx>dz)*dz + (dz>=dx)*dx;
  /// Normalized mass of electron in kg.
  const double m_e = arma::datum::m_e/m_i;
  /// Derived normalization constant of Alfven velocity in m/s. \f$V_0 = V_A = B_0 / \sqrt{\mu_0 n_0 m_i}\f$
  const double V_0 = B_0/std::sqrt(mu_0*n_0*m_i);
  /// Derived normalization constant of time in seconds.
  const double t_0 = l_0/V_0;
  /// Derived normalization constant of pressure in Pascals. \f$p_0 = B_0^2 / \mu_0\f$
  const double p_0 = B_0*B_0/mu_0;
  /// Derived normalization constant of temperature in Kelvin. \f$T_0 = p_0 / (n_0 k_b)\f$
  const double T_0 = p_0/(n_0*k_b);
  /// In normalized units, default is sun value of 0.27395 km/s/s.
  const double g = 0.27395e3*t_0/V_0;
  /// Derived normalization constant for heat transfer.
  const double q_0               = p_0*V_0;
  /// Derived normalization constant for charge in Coulombs.
  const double e_0               = B_0/(mu_0*n_0*V_0*l_0);
  /// Normalized elementary charge constant.
  const double q                 = e_/e_0;
  /// Derived normalization constant for heat transfer coefficient.
  const double kappa_0           = q_0*l_0/T_0;

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

  // Ranges
  const viennacl::range r_grid = viennacl::range(0, nx*nz);
  const viennacl::range r_Np = viennacl::range(n_p, n_p+1);
  const viennacl::range r_Nn = viennacl::range(n_n, n_n+1);
  const viennacl::range r_NVx = viennacl::range(Vx, Vx+1);
  const viennacl::range r_NVz = viennacl::range(Vz, Vz+1);
  const viennacl::range r_NVp = viennacl::range(Vp, Vp+1);
  const viennacl::range r_NUx = viennacl::range(Ux, Ux+1);
  const viennacl::range r_NUz = viennacl::range(Uz, Uz+1);
  const viennacl::range r_NUp = viennacl::range(Up, Up+1);
  const viennacl::range r_Ep = viennacl::range(E_p, E_p+1);
  const viennacl::range r_En = viennacl::range(E_n, E_n+1);
  const viennacl::range r_Bx = viennacl::range(Bx, Bx+1);
  const viennacl::range r_Bz = viennacl::range(Bz, Bz+1);
  const viennacl::range r_Bp = viennacl::range(Bp, Bp+1);
  const viennacl::range r_GLM = viennacl::range(GLM, GLM+1);
  const arma::uword num_of_eq = 14;

  Context(arma::uword _nx,
          arma::uword _nz,
          BoundaryCondition _bc_up=NEUMANN,
          BoundaryCondition _bc_down=NEUMANN,
          BoundaryCondition _bc_right=NEUMANN,
          BoundaryCondition _bc_left=NEUMANN,
          const Settings _settings = Settings()
          ) : nx(_nx), nz(_nz),
              bc_up(_bc_up),
              bc_down(_bc_down),
              bc_right(_bc_right),
              bc_left(_bc_left),
              settings(_settings) {
  }

  /*!
   * \brief returns number of grid points per unknown.
   * \return Number of grid points per unknown.
   */
  arma::uword num_of_grid() const { return nz*nx; }

  /*!
   * \brief Set \f$\Delta t\f$ in normalized units of \ref t_0 .
   * \param dt Time step in normalized units of \ref t_0 .
   * Since \ref m_dt is a private member. This variable must be changed through this function.
   */
  void set_dt(const double &dt) {
    m_dt = dt;
  }

  /*!
   * \brief Advance the state of the simulation.
   * This will advance the state of the simulation so that the \ref time_elapsed()
   * and \ref time_step() are updated.
   */
  void advance() {
    m_time_elapsed += m_dt;
    m_time_step++;
  }

  /// Getter of \f$\Delta t\f$
  double dt() const { return m_dt; }
  /// Getter of \f$t\f$
  double time_elapsed() const { return m_time_elapsed; }
  /// Getter of \f$n\f$
  uint time_step() const { return m_time_step; }

  // Matrix range returns
  inline viennacl::range range(const uint &var) const { return viennacl::range(var, var+1); }
  inline viennacl::matrix_range<vcl_mat> range(const uint &var, const vcl_mat &xn) {
    return project(xn, r_grid, range(var));
  }
  inline viennacl::matrix_range<vcl_mat> v_Np(const vcl_mat &xn) {
    return project(xn, r_grid, r_Np);
  }
  inline viennacl::matrix_range<vcl_mat> v_Nn(const vcl_mat &xn) {
    return project(xn, r_grid, r_Nn);
  }
  inline viennacl::matrix_range<vcl_mat> v_NVx(const vcl_mat &xn) {
    return project(xn, r_grid, r_NVx);
  }
  inline viennacl::matrix_range<vcl_mat> v_NVz(const vcl_mat &xn) {
    return project(xn, r_grid, r_NVz);
  }
  inline viennacl::matrix_range<vcl_mat> v_NVp(const vcl_mat &xn) {
    return project(xn, r_grid, r_NVp);
  }
  inline viennacl::matrix_range<vcl_mat> v_NUx(const vcl_mat &xn) {
    return project(xn, r_grid, r_NUx);
  }
  inline viennacl::matrix_range<vcl_mat> v_NUz(const vcl_mat &xn) {
    return project(xn, r_grid, r_NUz);
  }
  inline viennacl::matrix_range<vcl_mat> v_NUp(const vcl_mat &xn) {
    return project(xn, r_grid, r_NUp);
  }
  inline viennacl::matrix_range<vcl_mat> v_Bx(const vcl_mat &xn) {
    return project(xn, r_grid, r_Bx);
  }
  inline viennacl::matrix_range<vcl_mat> v_Bz(const vcl_mat &xn) {
    return project(xn, r_grid, r_Bz);
  }
  inline viennacl::matrix_range<vcl_mat> v_Bp(const vcl_mat &xn) {
    return project(xn, r_grid, r_Bp);
  }
  inline viennacl::matrix_range<vcl_mat> v_GLM(const vcl_mat &xn) {
    return project(xn, r_grid, r_GLM);
  }
  inline viennacl::matrix_range<vcl_mat> v_Ep(const vcl_mat &xn) {
    return project(xn, r_grid, r_Ep);
  }
  inline viennacl::matrix_range<vcl_mat> v_En(const vcl_mat &xn) {
    return project(xn, r_grid, r_En);
  }
};

} // namespace types

/*!
 * \brief Functions to build or apply the 'operators'.
 * Permutations of the matrix.
 */
namespace operators {

/*!
* \brief Permutate a solution down one cell.
* \param xn Solution matrix context.
* \param ctx Simulation context.
* \return Matrix of \f$A_{j-1}\f$
* Every index is replaced by the cell below it while maintaining boundary conditions.
*/
vcl_mat jm1(const vcl_mat &xn, comfi::types::Context ctx);

/*!
* \brief Permutate a solution up one cell.
* \param xn Solution matrix context.
* \param ctx Simulation context.
* \return Matrix of \f$A_{j+1}\f$
* Every index is replaced by the cell above it while maintaining boundary conditions.
*/
vcl_mat jp1(const vcl_mat &xn, comfi::types::Context ctx);

/*!
* \brief Permutate a solution left one cell.
* \param xn Solution matrix context.
* \param ctx Simulation context.
* \return Matrix of \f$A_{i-1}\f$
* Every index is replaced by the cell to the left of it while maintaining boundary conditions.
*/
vcl_mat im1(const vcl_mat &xn, comfi::types::Context ctx);

/*!
* \brief Permutate a solution right one cell.
* \param xn Solution matrix context.
* \param ctx Simulation context.
* \return Matrix of \f$A_{i+1}\f$
* Every index is replaced by the cell to the right of it while maintaining boundary conditions.
*/
vcl_mat ip1(const vcl_mat &xn, comfi::types::Context ctx);

/*!
 * \brief Permutate a result arma::vector right one cell.
 * \param ctx Simulation context.
 * \return Sparse matrix to multiply a result arma::vector by.
 * Every index is replaced by the cell to the right of it while maintaining boundary conditions.
 */
const arma::sp_mat buildPip1(comfi::types::Context &ctx);

/*!
 * \brief Permutate a result arma::vector left one cell.
 * \param ctx Simulation context.
 * \return Sparse matrix to multiply a result arma::vector by.
 * Every index is replaced by the cell to the left of it while maintaining boundary conditions.
 */
const arma::sp_mat buildPim1(comfi::types::Context &ctx);

} // namespace operators

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
 * \brief save_solution Saves the solution to the output folder
 * \param x0 Solution matrix.
 * \param ctx Simulation context.
 * \param data_name data set name to save it under in hdf5 tree. Defaults to "unknowns"
 * \return Success or fail.
 */
bool save_solution(const vcl_mat &x0,
                   comfi::types::Context &ctx,
                   const std::string &data_name = "unknowns");

/*!
 * \brief sendtolog Send message to log file.
 * \param message Message std::string to show in log file.
 * \param filename The log filename make sure this is always the same.
 */
void sendtolog(const std::string message, const std::string filename);

/*!
 * \brief Calculate the initial condition based on Orszang-Tang Vortex.
 * \param ctx Simulation context
 * \return initial condition matrix
 */
vcl_mat ot_vortex_ic(comfi::types::Context &ctx);

/*!
 * \brief Calculate the initial condition based on Sod's Shock Tube.
 * \param ctx Simulation context
 * \return initial condition matrix
 */
vcl_mat shock_tube_ic(comfi::types::Context &ctx);

// Misc

/*!
 * \brief getmaxV get fast mode speed + local speed OR resistivity speed whichever is faster
 * \param x0 Solution matrix
 * \param ctx Simulation context
 * \return Max characteristic speed in normalized units
 */
double getmaxV(const vcl_mat &x0, comfi::types::Context &ctx);

/*!
 * \brief vec_to_mat Change a column vector to a viennacl::matrix type with one column
 * \param vec viennacl::vector
 * \return viennacl::matrix of one column
 */
vcl_mat vec_to_mat(const vcl_vec &vec);

/*!
 * \brief Write a binary file of a field arma::vector. Seperate binary files for each direction.
 * \param x0 result arma::vector
 * \param name Name of field, will be used as filename prefix.
 * \param timestep Time step to save as
 * \return True if no error.
 */
bool saveField(const arma::vec &x0, const std::string name, const int timestep);

/*!
 * \brief Read the arguments passed to the executable and change the Settings struct.
 */
void interpret_arguments(comfi::types::Settings &settings, int argc, char** argv);

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

/*!
 * Linear algebra routines and deriving variables and such are all found here.
 */
namespace routines
{

/*!
 * \brief Build an eigenvalue matrix (wave speeds) for the Lax-Friedrichs scheme.
 * \param p_eig Ion wave speeds.
 * \param n_eig Neutral wave speeds.
 */
vcl_mat build_eig_matrix(const vcl_mat &p_eig, const vcl_mat &n_eig, comfi::types::Context &ctx);

/*!
 * \brief pressure_n Returns the plasma pressure based on the type of energy being used
 * \param xn Solution matrix
 * \param ctx Simulation context
 * \return Simulation pressure matrix in (grid, 1) dimensions
 */
vcl_mat pressure_p(const vcl_mat &xn, comfi::types::Context &ctx);

/*!
 * \brief pressure_n Returns the neutral pressure based on the type of energy being used
 * \param xn Solution matrix
 * \param ctx Simulation context
 * \return Simulation pressure matrix in (grid, 1) dimensions
 */
vcl_mat pressure_n(const vcl_mat &xn, comfi::types::Context &ctx);

/*!
 * \brief Fx Get flux values in the x direction.
 * \param xn Solution matrix in the cell edge.
 * \param xn_ij Solution matrix in the cell center.
 * \param ctx Simulation context.
 * \return Flux function matrix results.
 */
vcl_mat Fx(const vcl_mat &xn, const vcl_mat &xn_ij, comfi::types::Context &ctx);

/*!
 * \brief Fz Get flux values in the z direction.
 * \param xn Solution matrix in the cell edge.
 * \param xn_ij Solution matrix in the cell center.
 * \param ctx Simulation context.
 * \return Flux function matrix results.
 */
vcl_mat Fz(const vcl_mat &xn, const vcl_mat &xn_ij, comfi::types::Context &ctx);

/*!
 * \brief Re_MUSCL Do flux reconstruction with the MUSCL scheme and any source terms.
 * \param xn Solution matrix
 * \param ctx Simulation context
 * \return Solution.
 */
vcl_mat Re_MUSCL(const vcl_mat &xn, comfi::types::Context &ctx);

/*!
 * \brief Flux limiter
 * \param r ratio of gradients
 * \return Limiter function values
 */
vcl_mat fluxl(const vcl_mat &r);

/*!
 * \brief Get neutral sound speed
 * \param xn matrix of results
 * \param ctx Simulation context
 * \return vector (vcl_mat) of neutral sound speed
 */
vcl_mat sound_speed_neutral_mat(const vcl_mat &xn, comfi::types::Context &ctx);

/*!
 * \brief Get neutral sound speed
 * \param xn matrix of results
 * \param ctx Simulation context
 * \return vector of neutral sound speed
 */
vcl_vec sound_speed_neutral(const vcl_mat &xn, const comfi::types::Context &ctx);

/*!
 * \brief fast_speed_z Get vertical fast mode speed
 * \param xn result matrix
 * \param ctx Simulation context.
 * \return Column vector vcl_mat of vertical fast mode speed of ions.
 */
vcl_mat fast_speed_z_mat(const vcl_mat &xn, comfi::types::Context &ctx);

/*!
 * \brief fast_speed_z Get vertical fast mode speed
 * \param xn result matrix
 * \param ctx Simulation context.
 * \return Column vector of vertical fast mode speed of ions.
 */
vcl_vec fast_speed_z(const vcl_mat &xn, comfi::types::Context &ctx);

/*!
 * \brief fast_speed_x Get horizontal fast mode speed
 * \param xn Result matrix.
 * \param ctx	Simulation context.
 * \return Column vector (vcl_mat) of horizontal fast mode speeds.
 */
vcl_mat fast_speed_x_mat(const vcl_mat &xn, comfi::types::Context &ctx);

/*!
 * \brief fast_speed_x Get horizontal fast mode speed
 * \param xn Result matrix.
 * \param ctx	Simulation context.
 * \return Column vector of horizontal fast mode speeds.
 */
vcl_vec fast_speed_x(const vcl_mat &xn, comfi::types::Context &ctx);

/*!
 * \brief polyval compute's polynomial using Herner's scheme
 * \param p polynomial coefficient vector of size (polynomial degree + 1)
 * \param x vector to be computed
 * \return result of the polynomial
 */
vcl_vec polyval(const arma::vec &p, const vcl_vec &x);

/*!
 * \brief computeRHS_Euler Compute right hand side using Eulerian time stepping.
 * \param xn Current time step result matrix
 * \param ctx Simulation context
 * \return Right hand side result solution matrix
 */
vcl_mat computeRHS_Euler(const vcl_mat &xn, comfi::types::Context &ctx);

/*!
 * \brief computeRHS_RK4 Compute right hand side using Runge-Kutta 4 time stepping.
 * \param xn Current time step result arma::vector
 * \param dt Change in time
 * \param t Time elapsed
 * \param op operators
 * \param bg Background data
 * \return  Right hand side result arma::vector
 */
vcl_mat computeRHS_RK4(const vcl_mat &xn, comfi::types::Context &ctx);

/*!
 * \brief Shock Tube dirichlet boundary conditions
 * \param Lxn Left state
 * \param Rxn Right state
 * \param ctx Context
 */
void topbc_shock_tube(vcl_mat &Lxn, vcl_mat &Rxn, comfi::types::Context &ctx);

/*!
 * \brief Shock Tube dirichlet boundary conditions
 * \param Lxn Left state
 * \param Rxn Right state
 * \param ctx Context
 */
void bottombc_shock_tube(vcl_mat &Lxn, vcl_mat &Rxn, comfi::types::Context &ctx);

// inlines

/* inline const vcl_vec div_tvd(const vcl_vec &s_iph, const vcl_vec &s_imh, const vcl_vec &s_jph, const vcl_vec &s_jmh) */
/* { */
/*   const vcl_vec xpart = (s_iph-s_imh)/dx; */
/*   const vcl_vec zpart = (s_jph-s_jmh)/dz; */
/*   return xpart+zpart; */
/* } */

} // namespace routines

/*!
 * Solar chromosphere math functions.
 */
namespace sol
{

inline double nu_nn(const double &nn, const double &T, const comfi::types::Context &ctx)
{
  const double sigma_nn = 7.73e-19; //m-2
  const double m_nn = 1.007825*arma::datum::m_u;
  const double nn_coeff = sigma_nn * std::sqrt(16.0*arma::datum::k/(arma::datum::pi*m_nn));
  return (nn_coeff * nn * ctx.n_0 * std::sqrt(std::abs(T*ctx.T_0)) * ctx.t_0); // ion-neutral collision rate
}

inline arma::vec nu_nn(const arma::vec &nn, const arma::vec &T, const comfi::types::Context &ctx)
{
  const double sigma_nn = 7.73e-19; //m-2
  const double m_nn = 1.007825*arma::datum::m_u;
  const double nn_coeff = sigma_nn * std::sqrt(16.0*arma::datum::k/(arma::datum::pi*m_nn));
  return (nn_coeff * ctx.n_0 * ctx.t_0 * nn % arma::sqrt(arma::abs(T*ctx.T_0)) ); // ion-neutral collision rate
}

inline vcl_vec nu_nn(const vcl_vec &nn, const vcl_vec &T, const comfi::types::Context &ctx)
{
  const double sigma_nn = 7.73e-19; //m-2
  const double m_nn = 1.007825*arma::datum::m_u;
  const double nn_coeff = sigma_nn * std::sqrt(16.0*arma::datum::k/(arma::datum::pi*m_nn));
  return nn_coeff * ctx.n_0 * ctx.t_0  * viennacl::linalg::element_prod(nn, viennacl::linalg::element_sqrt(T*ctx.T_0)); // ion-neutral collision rate
}

inline double nu_in(const double &nn, const double &T, const comfi::types::Context &ctx)
{
  const double sigma_in = 1.16e-18; //m-2
  const double m_in = 1.007276466879*1.007825/(1.007276466879+1.007825);
  const double in_coeff = sigma_in * std::sqrt(8.0*arma::datum::k/(arma::datum::pi*m_in));
  return (in_coeff * nn * ctx.n_0 * std::sqrt(std::abs(T*ctx.T_0)) * ctx.t_0); // ion-neutral collision rate
}

inline arma::vec nu_in(const arma::vec &nn, const arma::vec &T, const comfi::types::Context &ctx)
{
  const double sigma_in = 1.16e-18; //m-2
  const double m_in = 1.007276466879*1.007825/(1.007276466879+1.007825);
  const double in_coeff = sigma_in * std::sqrt(8.0*arma::datum::k/(arma::datum::pi*m_in));
  return (in_coeff * ctx.n_0 * ctx.t_0 * nn % arma::sqrt(arma::abs(T*ctx.T_0)) ); // ion-neutral collision rate
}

inline vcl_vec nu_in(const vcl_vec &nn, const vcl_vec &T, const comfi::types::Context &ctx)
{
  const double sigma_in = 1.16e-18; //m-2
  const double m_in = 1.007276466879*1.007825/(1.007276466879+1.007825);
  const double in_coeff = sigma_in * std::sqrt(8.0*arma::datum::k/(arma::datum::pi*m_in));
  return in_coeff * ctx.n_0 * ctx.t_0  * viennacl::linalg::element_prod(nn, viennacl::linalg::element_sqrt(T*ctx.T_0)); // ion-neutral collision rate
}

inline double nu_en(const double &nn, const double &T, const comfi::types::Context &ctx)
{
  const double sigma_en = 1.0e-19; //m-2
  const double m_en = arma::datum::m_e*1.007825*arma::datum::m_u/(arma::datum::m_e+1.007825*arma::datum::m_u);
  const double en_coeff = sigma_en * std::sqrt(8.0*arma::datum::k/(arma::datum::pi*m_en));
  return (en_coeff * nn * ctx.n_0 * sqrt(std::abs(T*ctx.T_0)) * ctx.t_0); // electron-neutral collision rate
}

inline vcl_vec nu_en(const vcl_vec &nn, const vcl_vec &T, const comfi::types::Context &ctx)
{
  const double sigma_en = 1.0e-19; //m-2
  const double m_en = arma::datum::m_e*1.007825*arma::datum::m_u/(arma::datum::m_e+1.007825*arma::datum::m_u);
  const double en_coeff = sigma_en * std::sqrt(8.0*arma::datum::k/(arma::datum::pi*m_en));
  return en_coeff * ctx.n_0 * ctx.t_0 * viennacl::linalg::element_prod(nn, viennacl::linalg::element_sqrt(T*ctx.T_0)); // electron-neutral collision rate
}

inline double nu_ei(const double &ne, const double &T, const comfi::types::Context &ctx)
{
  const double coloumb_logarithm = 10.0;
  const double r = arma::datum::ec*arma::datum::ec/(4.0*arma::datum::pi*arma::datum::eps_0*arma::datum::k);
  const double sigma_ei = coloumb_logarithm*arma::datum::pi*r*r;
  const double coeff_ei = (4.0/3.0) * sigma_ei * std::sqrt(2.0*arma::datum::k/(arma::datum::pi*arma::datum::m_e));
  return (coeff_ei * ne * ctx.n_0 * std::pow(std::abs(T*ctx.T_0),-1.5) * ctx.t_0); // electron-ion collision rate
}

inline vcl_vec nu_ei(const vcl_vec &ne, const vcl_vec &Te, const comfi::types::Context &ctx)
{
  const double coloumb_logarithm = 10.0;
  const double r = arma::datum::ec*arma::datum::ec/(4.0*arma::datum::pi*arma::datum::eps_0*arma::datum::k);
  const double sigma_ei = coloumb_logarithm*arma::datum::pi*r*r;
  const double coeff_ei = (4.0/3.0) * sigma_ei * std::sqrt(2.0*arma::datum::k/(arma::datum::pi*arma::datum::m_e));
  return coeff_ei * ctx.n_0 * ctx.t_0* viennacl::linalg::element_prod(ne,viennacl::linalg::element_exp(-1.5*viennacl::linalg::element_log(ctx.T_0*Te))); // electron-ion collision rate
}

inline vcl_vec nu_ii(const vcl_vec &Np, const vcl_vec &Tp, const comfi::types::Context &ctx)
{
  const double coloumb_logarithm = 10.0;
  const double r = arma::datum::ec*arma::datum::ec/(4.0*arma::datum::pi*arma::datum::eps_0*arma::datum::k); //without T
  const double sigma_ii = coloumb_logarithm*arma::datum::pi*r*r;
  const double coeff_ii = (4.0/3.0) * sigma_ii * std::sqrt(2.0*arma::datum::k/(arma::datum::pi*arma::datum::m_p));
  return ctx.t_0 * ctx.n_0 * coeff_ii * viennacl::linalg::element_prod(Np, viennacl::linalg::element_exp(-1.5*viennacl::linalg::element_log(ctx.T_0*Tp))); // electron-ion collision rate
}

inline double nu_ii(const double &Np, const double &Tp, const comfi::types::Context &ctx)
{
  const double coloumb_logarithm = 10.0;
  const double r = arma::datum::ec*arma::datum::ec/(4.0*arma::datum::pi*arma::datum::eps_0*arma::datum::k); //without T
  const double sigma_ii = coloumb_logarithm*arma::datum::pi*r*r;
  const double coeff_ii = (4.0/3.0) * sigma_ii * std::sqrt(2.0*arma::datum::k/(arma::datum::pi*arma::datum::m_p));
  return coeff_ii * ctx.t_0 * Np * ctx.n_0 * std::exp(-1.5*std::log(ctx.T_0*Tp)); // electron-ion collision rate
}

inline vcl_vec resistivity(const vcl_vec &Np, const vcl_vec &Nn, const vcl_vec &Tp, const vcl_vec &Tn, const comfi::types::Context &ctx)
{
  return (ctx.m_e/(ctx.q*ctx.q))*viennacl::linalg::element_div((nu_ei(Np, Tp, ctx)+nu_en(Nn, 0.5*(Tp+Tn), ctx)),Np);
}
inline double resistivity(const double &Np, const double &Nn, const double &Tp, const double &Tn, const comfi::types::Context &ctx)
{
  return (ctx.m_e/(ctx.q*ctx.q)) * (nu_ei(Np, Tp, ctx)+nu_en(Nn, 0.5*(Tp+Tn), ctx))/Np;
}

inline double thermalvel(const double &T, const comfi::types::Context &ctx)
{
  return std::sqrt(2.0*arma::datum::k/arma::datum::m_p) * std::sqrt(T*ctx.T_0) / ctx.V_0;
}

inline vcl_vec thermalvel(const vcl_vec &T, const comfi::types::Context &ctx)
{
  using namespace viennacl::linalg;
  return std::sqrt(2.0*arma::datum::k/arma::datum::m_p) * element_sqrt(T*ctx.T_0) / ctx.V_0;
}

inline double kappa_n(const double &Tp, const double &Tn, const double &Np, const double &Nn, const comfi::types::Context &ctx)
{
  return Nn * thermalvel(Tn, ctx) * thermalvel(Tn, ctx) / nu_nn(Nn, Tn, ctx);
}

inline vcl_vec kappa_n(const vcl_vec &Tp, const vcl_vec &Tn, const vcl_vec &Np, const vcl_vec &Nn, const comfi::types::Context &ctx)
{
  using namespace viennacl::linalg;
  const vcl_vec v2 = element_prod(thermalvel(Tn, ctx), thermalvel(Tn, ctx));
  const vcl_vec nv2 = element_prod(Nn, v2);
  const vcl_vec nunn = nu_nn(Nn, Tn, ctx);
  return element_div(nv2, nunn);
}

inline double kappa_p(const double &Tp, const double &Tn, const double &Np, const double &Nn, const comfi::types::Context &ctx)
{
  return Np * thermalvel(Tp, ctx) * thermalvel(Tp, ctx) / (nu_in(Nn, 0.5*(Tp+Tn), ctx) + nu_ii(Np, Tp, ctx));
}

inline vcl_vec kappa_p(const vcl_vec &Tp, const vcl_vec &Tn, const vcl_vec &Np, const vcl_vec &Nn, const comfi::types::Context &ctx)
{
  using namespace viennacl::linalg;
  const vcl_vec v2 = element_prod(thermalvel(Tp, ctx), thermalvel(Tp, ctx));
  const vcl_vec nv2 = element_prod(Np, v2);
  const vcl_vec nuin = nu_in(Nn, 0.5*(Tp+Tn), ctx);
  const vcl_vec nuii = nu_ii(Np, Tp, ctx);
  return element_div(nv2, nuin+nuii);
}

/*!
 * \brief Calculate recombination coefficient for ions to recombine to neutrals.
 * \param Tp Ion temperature
 * \return Vector of normalized volumetric recombination rate.
 */
inline vcl_vec recomb_coeff(const vcl_vec &T, const comfi::types::Context &ctx)
{
  using namespace viennacl::linalg;
  static const vcl_vec one = viennacl::scalar_vector<double>(T.size(), 1.0);
  const vcl_vec beta = 0.6 * 13.6 * 1.6021766208e-19 * element_div(one, arma::datum::k*ctx.T_0*T);
  vcl_vec coeff = 0.4288*one + 0.5*element_log(beta) + 0.4698*element_pow(beta, -(1.0/3.0)*one);
  coeff = element_prod(5.20e-20*element_sqrt(beta), coeff);
  return ctx.t_0 * ctx.n_0 * coeff;
}

inline double recomb_coeff(const double &T, const comfi::types::Context &ctx)
{
  const double beta = 0.6 * 13.6 * 1.6021766208e-19 / (arma::datum::k*ctx.T_0*T);
  return ctx.t_0 * ctx.n_0 * 5.20e-20 * std::sqrt(beta) * (0.4288+std::log(beta)+0.4698*std::pow(beta, -(1.0/3.0)));
}

/*!
 * \brief Calculate normalized ionization coefficient for neutrals to ionize due to collisions (Draine p 134)
 * \param Tn Neutral temperature
 * \return Vector of normalized volumetric ionization rate
 */
inline vcl_vec ionization_coeff(const vcl_vec &T, const comfi::types::Context &ctx)
{
  using namespace viennacl::linalg;
  static const vcl_vec one = viennacl::scalar_vector<double>(T.size(), 1.0);
  const vcl_vec beta = 0.6 * 13.6 * 1.6021766208e-19 * element_div(one, arma::datum::k*ctx.T_0*T);
  const vcl_vec coeff = element_div(2.34e-14*element_exp(-beta), element_sqrt(beta));
  return ctx.t_0 * ctx.n_0 * coeff;

}

inline double ionization_coeff(const double &T, const comfi::types::Context &ctx)
{
  const double beta = 0.6 * 13.6 * 1.6021766208e-19 / (arma::datum::k*ctx.T_0*T);
  return ctx.t_0 * ctx.n_0 * 2.34e-14 * std::exp(-beta) / std::sqrt(beta);
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
  return i+j*ctx.nx;
}

/*
vim: tabstop=2
vim: shiftwidth=2
vim: smarttab
vim: expandtab
*/
