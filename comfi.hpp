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
typedef viennacl::matrix<double, viennacl::column_major> vcl_mat; /*!< GPU dense matrix with ViennaCL */

// local includes
#include "params.h"
#include "fluxl.h"

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
 * \brief The Boundary Condition enum.
 */
enum BoundaryCondition {PERIODIC, /*!< Periodic boundary condition (i+1 at right boundary is i=0) */
                        DIRICHLET, /*!< Dirichlet boundary conditions. Inserting data up to user. */
                        NEUMANN, /*!< Neumann boundary conditions. (i+1 at right boundary = i) */
                        MIRROR, /*!< Mirror boundary conditions. (i+1 at right boundary = i or negative that value in case of x component) */
                        DIMENSIONLESS /*!< This boundary is not in a free dimension. */
                       };
/*!
 * \struct BgData
 * \brief Background Data is stored here.
 * Background Data data structure which has cpu matrices for background, Bz, Np, Nn
*/
struct BgData {
  /*!
     * \brief BgData Background data is stored here.
     * \param BBzfile Vertical magnetic field data in a csv text file
     * \param BNpfile Ion density data in a csv text file
     * \param BNnfile Neutral density data in a csv text file
     */
    BgData(std::string BBzfile     = "input/BBz.csv",
           std::string BNpfile     = "input/BNp.csv",
           std::string BNnfile     = "input/BNn.csv");

    BgData(arma::vec _BBz, arma::mat _BNp, arma::mat _BNn)
      : BBz(_BBz), BNp(_BNp), BNn(_BNn) {}

    BgData(arma::vec _BBz, arma::vec _BBx, arma::mat _BNp, arma::mat _BNn)
      : BBz(_BBz), BBx(_BBx), BNp(_BNp), BNn(_BNn) {}
    // members
    arma::vec BBz     = arma::zeros<arma::vec>(nx); /*!< Background Magnetic Field vertical */
    arma::vec BBx     = arma::zeros<arma::vec>(nz); /*!< Background Magnetic Field vertical */
    arma::mat BNp     = arma::zeros<arma::mat>(nz, nx); /*!< Background Ion Density */
    arma::mat BNn     = arma::zeros<arma::mat>(nz, nx); /*!< Background Neutral Density */
};

/*!
 * \brief Data structure with sparse matrices that act like operators.
 */
struct Operators{
  /*!
   * \brief Sparse matrix operators data struct.
   * \param _LeftBC Left boundary condition (i<0)
   * \param _RightBC Right boundary condition (i>nx)
   * \param _UpBC Top boundary condition (j>nz)
   * \param _DownBC Bottom boundary condition (j<0)
   */
  Operators(const BoundaryCondition _LeftBC = MIRROR,
            const BoundaryCondition _RightBC = MIRROR,
            const BoundaryCondition _UpBC = NEUMANN,
            const BoundaryCondition _DownBC = NEUMANN
            );

  /*!
   * \brief Left boundary condition i=0
   * \return Left boundary condition
   */
  BoundaryCondition getLeftBC() const ;

  /*!
   * \brief Right boundary condition i=x_size-1
   * \return Right boundary condition
   */
  BoundaryCondition getRightBC() const ;

  /*!
   * \brief Upper boundary condition j=z_size-1
   * \return Upper boundary condition
   */
  BoundaryCondition getUpBC() const ;

  /*!
   * \brief Bottom boundary condition j=0
   * \return Bottom boundary
   */
  BoundaryCondition getDownBC() const ;

  vcl_sp_mat Pjp1, /*!< Permutate result vector up one (j+1) see: \ref mhdsim::operators::buildPjp1()  */
             Pjm1, /*!< Permutate result vector down one (j-1) see: \ref mhdsim::operators::buildPjm1()   */
             Pip1, /*!< Permutate result vector right one (i+1)  see: \ref mhdsim::operators::buildPip1()  */
             Pim1, /*!< Permutate result vector left one (i-1)  see: \ref mhdsim::operators::buildPim1()  */
             Pip2, /*!< Permutate result vector left one (i+2)  see: \ref mhdsim::operators::buildPip2()  */
             Pim2, /*!< Permutate result vector left one (i-2)  see: \ref mhdsim::operators::buildPim2()  */
             Pjp2, /*!< Permutate result vector left one (j+2)  see: \ref mhdsim::operators::buildPjp2()  */
             Pjm2; /*!< Permutate result vector left one (j-2)  see: \ref mhdsim::operators::buildPjm2()  */
  vcl_sp_mat SG; /*!< Multiply densities by acceleration due to gravity and index in momentum. see: \ref mhdsim::operators::buildSG() */
  vcl_sp_mat PEVx, /*!< Place local speed eigenvalues (momentum before division by density) in the indeces used in \ref Fx . see: \ref mhdsim::operators::buildPEigVx() */
             PEVz, /*!< Place local speed eigenvalues (momentum before division by density) in the indeces used in \ref Fz . see: \ref mhdsim::operators::buildPEigVz() */
             PN; /*!< Place densities in the indeces used in \ref Fz and \ref Fx . see: \ref mhdsim::operators::buildPN() */
  vcl_sp_mat Bf, /*!< Sparse matrix multiply with a result vector to get a magnetic field of field vector type. see \ref mhdsim::operators::buildBfield() */
             Uf, /*!< Sparse matrix multiply with a result vector to get a neutral momentum field of field vector type. see \ref mhdsim::operators::buildUfield() */
             Vf; /*!< Sparse matrix multiply with a result vector to get a ion momentum field of field vector type. see \ref mhdsim::operators::buildVfield() */
  vcl_sp_mat pFBB; /*!< Sparse matrix to multiply by a result vector and get magnetic field values in ion momentum indeces for the first multiple of magnetic tension. see: \ref mhdsim::operators::buildpFBB() */
  vcl_sp_mat FxVB; /*!< Sparse matrix to multiply a result vector and get the first term in the flux VB-BV in the x direction. see: \ref mhdsim::operators::buildFxVB() */
  vcl_sp_mat FzVB; /*!< Sparse matrix to multiply a result vector and get the first term in the flux VB-BV in the z direction. see: \ref mhdsim::operators::buildFzVB() */
  vcl_sp_mat BottomBC; /*!< Empty matrix except for ones at the bottom values (z=0) used for dirichlet boundary conditions. see: \ref mhdsim::operators::buildBottomBC() */
  vcl_sp_mat I; /*!< Identity matrix for result vectors. */
  vcl_sp_mat ImBottom, /*!< Result of \ref I - \ref BottomBC . */
             ImTop, /*!< Result of \ref I - \ref TopBC . */
             ImGLM; /*!< Result of \ref I - GLM indeces. */
  vcl_sp_mat fdotx, /*!< Sparse matrix to multiply field vector to get a scalar vector with the x components */
             fdotz, /*!< Sparse matrix to multiply field vector to get a scalar vector with the z components */
             fdotp; /*!< Sparse matrix to multiply field vector to get a scalar vector with the perpendicular component */
  vcl_sp_mat s2f, /*!< Sparse matrix to multiply scalar vector to field vector with the scalar value for all components. see: \ref mhdsim::operators::field_scalar2field() */
             s2xf, /*!< Sparse matrix to multiply scalar vector to field vector with the scalar value for all components. see: \ref mhdsim::operators::field_scalar2xfield() */
             s2pf, /*!< Sparse matrix to multiply scalar vector to field vector with the scalar value for all components. see: \ref mhdsim::operators::field_scalar2pfield() */
             s2zf; /*!< Sparse matrix to multiply scalar vector to field vector with the scalar value for all components. see: \ref mhdsim::operators::field_scalar2zfield() */
  vcl_sp_mat cross1; /*!< Sparse matrix to multiply field vector to get first term of a cross product. */
  vcl_sp_mat cross2; /*!< Sparse matrix to multiply a field vector to get second term to subtract for cross product. */
  vcl_sp_mat curl; /*!< Sparse matrix to multiply a field vector to get a centered difference curl. */
  vcl_sp_mat s2GLM, /*!< Sparse matrix to multiply scalar vector to get a result vector with scalar values in GLM indeces. see: \ref mhdsim::operators::builds2GLM() */
             s2Np, /*!< Sparse matrix to multiply scalar vector to get a result vector with scalar values in ion density indeces. see: \ref mhdsim::operators::builds2Np() */
             s2Nn, /*!< Sparse matrix to multiply scalar vector to get a result vector with scalar values in neutral density indeces. see: \ref mhdsim::operators::builds2Nn() */
             s2Tp, /*!< Sparse matrix to multiply scalar vector to get a result vector with scalar values in ion temperature indeces. see: \ref mhdsim::operators::builds2Tp() */
             s2Tn; /*!< Sparse matrix to multiply scalar vector to get a result vector with scalar values in neutral temperature indeces. see: \ref mhdsim::operators::builds2Tn() */
  vcl_sp_mat f2B, /*!< Sparse matrix to multiply a field vector to get a result vector with components of field in magnetic field indeces. see: \ref mhdsim::operators::buildf2B() */
             f2V, /*!< Sparse matrix to multiply a field vector to get a result vector with components of field in ion momentum field indeces. see: \ref mhdsim::operators::buildf2V() */
             f2U, /*!< Sparse matrix to multiply a field vector to get a result vector with components of field in neutral momentum field indeces. see: \ref mhdsim::operators::buildf2U() */
             f2s; /*!< Sparse matrix to multiply a field vector to get a scalar vector Ax+Az+Ap values. */
  vcl_sp_mat Nps, /*!< Sparse matrix to multiply result vector to get a scalar vector of ion density values. */
             Nns, /*!< Sparse matrix to multiply result vector to get a scalar vector of neutral density values. */
             Tps, /*!< Sparse matrix to multiply result vector to get a scalar vector of ion temperature values. */
             Tns, /*!< Sparse matrix to multiply result vector to get a scalar vector of neutral temperature values. */
             GLMs; /*!< Sparse matrix to multiply result vector to get a scalar vector of lagrange multiplier values. */
  vcl_sp_mat grad, /*!< Sparse matrix to multiply a scalar vector and get a centered difference gradient field vector. */
             div; /*!< Sparse matrix to multiply a field vector to get a centered difference divergence. */
  vcl_sp_mat s2Vx, /*!< Sparse matrix to multiply a scalar vector to get a result vector with values in ion momentum x indeces. */
             s2Vz, /*!< Sparse matrix to multiply a scalar vector to get a result vector with values in ion momentum z indeces. */
             s2Bx, /*!< Sparse matrix to multiply a scalar vector to get a result vector with values in magnetic field x indeces. */
             s2Bz, /*!< Sparse matrix to multiply a scalar vector to get a result vector with values in magnetic field z indeces. */
             s2Ux, /*!< Sparse matrix to multiply a scalar vector to get a result vector with values in neutral momentum x indeces. */
             s2Uz; /*!< Sparse matrix to multiply a scalar vector to get a result vector with values in neutral momentum z indeces. */
  double     ch = 0, /*!< Value of hyperbolic speed of lagrange multiplier */
             c2p = 0; /*!< Value of parabolic speed squared of lagrange multiplier */
  private:
  /*!
   * \brief Left boundary condition. Default \ref MIRROR .
   */
  BoundaryCondition LeftBC = MIRROR;

  /*!
   * \brief Right boundary condition. Default \ref MIRROR .
   */
  BoundaryCondition RightBC = MIRROR;

  /*!
   * \brief Top boundary condition. Default \ref NEUMANN .
   */
  BoundaryCondition UpBC = NEUMANN;

  /*!
   * \brief Bottom boundary condition. Default \ref NEUMANN .
   */
  BoundaryCondition DownBC = NEUMANN;
};

/*!
 * \brief The simulation context. Everything needed about the simulation is in here.
 */
class Context {
  const BoundaryCondition m_bc_up = NEUMANN,
                          m_bc_down = NEUMANN,
                          m_bc_right = NEUMANN,
                          m_bc_left = NEUMANN;
  const uint m_flags;
  const arma::uword m_nx, m_nz;
  const bool m_resumed = false;
  const arma::uword m_num_of_eq = 14;
  viennacl::range m_r_grid;
  viennacl::range m_r;
  viennacl::range m_r_Np;
  viennacl::range m_r_Nn;
  viennacl::range m_r_NVx;
  viennacl::range m_r_NVz;
  viennacl::range m_r_NVp;
  viennacl::range m_r_NUx;
  viennacl::range m_r_NUz;
  viennacl::range m_r_NUp;
  viennacl::range m_r_Ep;
  viennacl::range m_r_En;
  viennacl::range m_r_Bx;
  viennacl::range m_r_Bz;
  viennacl::range m_r_Bp;
  viennacl::range m_r_GLM;
  double m_dt = 0.0;
  double m_time_elapsed = 0.0;
  uint m_time_step = 0;

public:
  Context(arma::uword nx,
          arma::uword nz,
          BoundaryCondition bc_up=NEUMANN,
          BoundaryCondition bc_down=NEUMANN,
          BoundaryCondition bc_right=NEUMANN,
          BoundaryCondition bc_left=NEUMANN,
          uint flags = 0,
          bool resumed = false
          ) : m_nx(nx), m_nz(nz),
              m_bc_up(bc_up),
              m_bc_down(bc_down),
              m_bc_right(bc_right),
              m_bc_left(bc_left),
              m_flags(flags),
              m_resumed(resumed) {
  viennacl::range r_grid(0, m_nx*m_nz);
  m_r_grid = r_grid;
  viennacl::range r(0, 1);
  m_r = r;
  viennacl::range r_Np(n_p, n_p+1);
  m_r_Np = r_Np;
  viennacl::range r_Nn(n_n, n_n+1);
  m_r_Nn = r_Nn;
  viennacl::range r_Vx(Vx, Vx+1);
  m_r_NVx = r_Vx;
  viennacl::range r_Vz(Vz, Vz+1);
  m_r_NVz = r_Vz;
  viennacl::range r_Vp(Vp, Vp+1);
  m_r_NVp = r_Vp;
  viennacl::range r_Ux(Ux, Ux+1);
  m_r_NUx = r_Ux;
  viennacl::range r_Uz(Uz, Uz+1);
  m_r_NUz = r_Uz;
  viennacl::range r_Up(Up, Up+1);
  m_r_NUp = r_Up;
  viennacl::range r_Ep(E_p, E_p+1);
  m_r_Ep = r_Ep;
  viennacl::range r_En(E_n, E_n+1);
  m_r_En = r_En;
  viennacl::range r_Bx(Bx, Bx+1);
  m_r_Bx = r_Bx;
  viennacl::range r_Bz(Bz, Bz+1);
  m_r_Bz = r_Bz;
  viennacl::range r_Bp(Bp, Bp+1);
  m_r_Bp = r_Bp;
  viennacl::range r_GLM(GLM, GLM+1);
  m_r_GLM = r_GLM;
  }

  arma::uword nx() const { return m_nx; }
  arma::uword nz() const { return m_nz; }
  arma::uword num_of_grid() const { return m_nz*m_nx; }
  BoundaryCondition bc_up() const { return m_bc_up; }
  BoundaryCondition bc_down() const { return m_bc_down; }
  BoundaryCondition bc_right() const { return m_bc_right; }
  BoundaryCondition bc_left() const { return m_bc_left; }
  uint flags() const { return m_flags; }
  bool is_resumed() const  { return m_resumed; }
  arma::uword num_of_eq() const { return m_num_of_eq; }
  viennacl::range r() const { return m_r; }
  viennacl::range r_grid() const { return m_r_grid; }
  viennacl::range r_Np() const { return m_r_Np; }
  viennacl::range r_Nn() const { return m_r_Nn; }
  viennacl::range r_NVx() const { return m_r_NVx; }
  viennacl::range r_NVz() const { return m_r_NVz; }
  viennacl::range r_NVp() const { return m_r_NVp; }
  viennacl::range r_NUx() const { return m_r_NUx; }
  viennacl::range r_NUz() const { return m_r_NUz; }
  viennacl::range r_NUp() const { return m_r_NUp; }
  viennacl::range r_Bx() const { return m_r_Bx; }
  viennacl::range r_Bz() const { return m_r_Bz; }
  viennacl::range r_Bp() const { return m_r_Bp; }
  viennacl::range r_Ep() const { return m_r_Ep; }
  viennacl::range r_En() const { return m_r_En; }
  viennacl::range r_GLM() const { return m_r_GLM; }
  void set_dt(double dt, bool advance = true) {
    m_dt = dt;
    if (advance) {
      m_time_elapsed += dt;
      m_time_step++;
    }
  }
  double dt() const { return m_dt; }
  double time_elapsed() const { return m_time_elapsed; }
  uint time_step() const { return m_time_step; }
  // Matrix range returns
  inline viennacl::matrix_range<vcl_mat> v_Np(const vcl_mat &xn) {
    return project(xn, r_grid(), r_Np());
  }
  inline viennacl::matrix_range<vcl_mat> v_Nn(const vcl_mat &xn) {
    return project(xn, r_grid(), r_Nn());
  }
  inline viennacl::matrix_range<vcl_mat> v_NVx(const vcl_mat &xn) {
    return project(xn, r_grid(), r_NVx());
  }
  inline viennacl::matrix_range<vcl_mat> v_NVz(const vcl_mat &xn) {
    return project(xn, r_grid(), r_NVz());
  }
  inline viennacl::matrix_range<vcl_mat> v_NVp(const vcl_mat &xn) {
    return project(xn, r_grid(), r_NVp());
  }
  inline viennacl::matrix_range<vcl_mat> v_NUx(const vcl_mat &xn) {
    return project(xn, r_grid(), r_NUx());
  }
  inline viennacl::matrix_range<vcl_mat> v_NUz(const vcl_mat &xn) {
    return project(xn, r_grid(), r_NUz());
  }
  inline viennacl::matrix_range<vcl_mat> v_NUp(const vcl_mat &xn) {
    return project(xn, r_grid(), r_NUp());
  }
  inline viennacl::matrix_range<vcl_mat> v_Bx(const vcl_mat &xn) {
    return project(xn, r_grid(), r_Bx());
  }
  inline viennacl::matrix_range<vcl_mat> v_Bz(const vcl_mat &xn) {
    return project(xn, r_grid(), r_Bz());
  }
  inline viennacl::matrix_range<vcl_mat> v_Bp(const vcl_mat &xn) {
    return project(xn, r_grid(), r_Bp());
  }
  inline viennacl::matrix_range<vcl_mat> v_GLM(const vcl_mat &xn) {
    return project(xn, r_grid(), r_GLM());
  }
  inline viennacl::matrix_range<vcl_mat> v_Ep(const vcl_mat &xn) {
    return project(xn, r_grid(), r_Ep());
  }
  inline viennacl::matrix_range<vcl_mat> v_En(const vcl_mat &xn) {
    return project(xn, r_grid(), r_En());
  }
};

} // namespace types

/*!
 * Functions to build the 'operators' (sparse matrices to extract vectors) are found here. The sparse matrices typically reside in \ref mhdsim::types::Operators .
 * Functions that function as operators are here too.
 */
namespace operators {
/*!
* \brief Permutate a solution down one cell.
* \param xn Solution matrix context.
* \param ctx Simulation context.
* \return Matrix of j-1
* Every index is replaced by the cell below it while maintaining boundary conditions.
*/
vcl_mat jm1(const vcl_mat &xn, comfi::types::Context ctx);

/*!
* \brief Permutate a solution up one cell.
* \param xn Solution matrix context.
* \param ctx Simulation context.
* \return Matrix of j+1
* Every index is replaced by the cell above it while maintaining boundary conditions.
*/
vcl_mat jp1(const vcl_mat &xn, comfi::types::Context ctx);

/*!
* \brief Permutate a solution left one cell.
* \param xn Solution matrix context.
* \param ctx Simulation context.
* \return Matrix of i-1
* Every index is replaced by the cell to the left of it while maintaining boundary conditions.
*/
vcl_mat im1(const vcl_mat &xn, comfi::types::Context ctx);

/*!
* \brief Permutate a solution right one cell.
* \param xn Solution matrix context.
* \param ctx Simulation context.
* \return Matrix of i+1
* Every index is replaced by the cell to the right of it while maintaining boundary conditions.
*/
vcl_mat ip1(const vcl_mat &xn, comfi::types::Context ctx);

/*!
* \brief Permutate a result arma::vector up one cell.
* \param BC Upper boundary condition
* \return Sparse matrix to multiply a result arma::vector by.
* Every index is replaced by the cell above it while maintaining boundary conditions.
*/
const arma::sp_mat buildPjp1(comfi::types::BoundaryCondition BC);

/*!
 * \brief Permutate a result arma::vector down one cell.
 * \param BC Bottom boundary condition
 * \return Sparse matrix to multiply a result arma::vector by.
 * Every index is replaced by the cell below it while maintaining boundary conditions.
 */
const arma::sp_mat buildPjm1(comfi::types::BoundaryCondition BC);

/*!
 * \brief Permutate a result arma::vector right one cell.
 * \param BC Right boundary condition
 * \return Sparse matrix to multiply a result arma::vector by.
 * Every index is replaced by the cell to the right of it while maintaining boundary conditions.
 */
const arma::sp_mat buildPip1(comfi::types::BoundaryCondition BC);

/*!
 * \brief Permutate a result arma::vector left one cell.
 * \param BC Left boundary condition
 * \return Sparse matrix to multiply a result arma::vector by.
 * Every index is replaced by the cell to the left of it while maintaining boundary conditions.
 */
const arma::sp_mat buildPim1(comfi::types::BoundaryCondition BC);

/*!
 * \brief Permutate a result arma::vector up two cell.
 * \param BC Top boundary condition
 * \return Sparse matrix to multiply a result arma::vector by.
 * Every index is replaced by the cell two above it while maintaining boundary conditions.
 */
const arma::sp_mat buildPjp2(comfi::types::BoundaryCondition BC);

/*!
 * \brief Permutate a result arma::vector down two cell.
 * \param BC Bottom boundary condition
 * \return Sparse matrix to multiply a result arma::vector by.
 * Every index is replaced by the cell two below it while maintaining boundary conditions.
 */
const arma::sp_mat buildPjm2(comfi::types::BoundaryCondition BC);

/*!
 * \brief Permutate a result arma::vector right two cells.
 * \param BC Right boundary condition
 * \return Sparse matrix to multiply a result arma::vector by.
 * Every index is replaced by the cell two to the right of it while maintaining boundary conditions.
 */
const arma::sp_mat buildPip2(comfi::types::BoundaryCondition BC);

/*!
 * \brief Permutate a result arma::vector left two cells.
 * \param BC Left boundary condition
 * \return Sparse matrix to multiply a result arma::vector by.
 * Every index is replaced by the cell two to the left of it while maintaining boundary conditions.
 */
const arma::sp_mat buildPim2(comfi::types::BoundaryCondition BC);

/*!
 * \brief Builds a sparse matrix when multiplied by a scalar vector leads to a result vector with values of horizontal magnetic field.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat builds2Bx();

/*!
 * \brief Builds a sparse matrix when multiplied by a scalar vector leads to a result vector with values of vertical magnetic field.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat builds2Bz();

/*!
 * \brief Builds a sparse matrix when multiplied by a scalar vector leads to a result vector with values of Lagrange Multiplier.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat builds2GLM();

/*!
 * \brief Builds a sparse matrix when multiplied by a scalar vector leads to a result vector with values of ion density.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat builds2Np();

/*!
 * \brief Builds a sparse matrix when multiplied by a scalar vector leads to a result vector with values of neutral density.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat builds2Nn();

/*!
 * \brief Builds a sparse matrix when multiplied by a scalar vector leads to a result vector with values of ion temperature.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat builds2Tp();

/*!
 * \brief Builds a sparse matrix when multiplied by a scalar vector leads to a result vector with values of neutral temperature.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat builds2Tn();

/*!
 * \brief Builds a sparse matrix when multiplied by a scalar vector leads to a result vector with values of horizontal ion momentum.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat builds2Vx();

/*!
 * \brief Builds a sparse matrix when multiplied by a scalar vector leads to a result vector with values of vertical ion momentum.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat builds2Vz();

/*!
 * \brief Builds a sparse matrix when multiplied by a scalar vector leads to a result vector with values of horizontal neutral momentum.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat builds2Ux();

/*!
 * \brief Builds a sparse matrix when multiplied by a scalar vector leads to a result vector with values of vertical neutral momentum.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat builds2Uz();

/*!
 * \brief Builds a sparse matrix when multiplied by a field vector leads to a result vector with values of magnetic field components.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildf2B();

/*!
 * \brief Builds a sparse matrix when multiplied by a field vector leads to a result vector with values of ion momentum field components.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildf2V();

/*!
 * \brief Builds a sparse matrix when multiplied by a field vector leads to a result vector with values of neutral momentum field components.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildf2U();

/*!
 * \brief Builds sparse matrix that outputs the first intermediate term of cross product in a field vector.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildCross1();

/*!
 * \brief Builds sparse matrix that outputs the second intermediate term of cross product in a field vector.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildCross2();

/*!
 * \brief x component of local speed eigenvalue for all indeces.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildPEigVx();

/*!
 * \brief z component of local speed eigenvalue for all indeces.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildPEigVz();

/*!
 * \brief Build a sparse matrix operator to extract a magnetic field vector from a result vector.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildBfield();

/*!
 * \brief Build a sparse matrix operator to extract a ion momentum field vector from a result vector.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildVfield();

/*!
 * \brief Build a sparse matrix operator to extract a neutral momentum field vector from a result vector.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildUfield();

/*!
 * \brief Centered difference curl operator (sparse matrix) to multiply by a field vector.
 * \param LeftBC Left boundary condition
 * \param RightBC Right boundary condition
 * \param UpBC Top boundary condition
 * \param DownBC Down boundary condition
 * \return Sparse matrix to multiply a field vector by.
 */
const arma::sp_mat buildCurl(comfi::types::BoundaryCondition LeftBC,
                             comfi::types::BoundaryCondition RightBC,
                             comfi::types::BoundaryCondition UpBC,
                             comfi::types::BoundaryCondition DownBC);

/*!
 * \brief Operator to multiply BBx and BBz (magnetic tension) to the momentum indeces for the flux.
 * \return Sparse matrix to multiply field vector by.
 */
const arma::sp_mat buildpFBB();

/*!
 * \brief Operator to multiply VBx to the magnetic field indeces for the flux.
 * \return Sparse matrix to multiply a field vector by.
 */
const arma::sp_mat buildFxVB();

/*!
 * \brief Operator to multiply VBz to the magnetic field indeces for the flux.
 * \return Sparse matrix to multiply a field vector by.
 */
const arma::sp_mat buildFzVB();

/*!
 * \brief Permutate density values for the eigenvalues made with buildPEigVx() and buildPEigVz()
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildPN();

/*!
 * \brief Sparse matrix to multiply density with gravitational acceleration.
 * \return Sparse matrix to multiply result vector by.
 */
const arma::sp_mat buildSG();

/*!
 * \brief Build sparse matrix with 1s on bottom boundary values that are supposed to be Dirichlet boundary condition. (Density, Temp, Perp Velocity)
 * \return Sparse matrix to subtract from identity matrix.
 */
const arma::sp_mat buildTopBC();

/*!
 * \brief Build sparse matrix with 1s on bottom boundary values that are supposed to be Dirichlet boundary condition. (Density, Temp, Perp Velocity)
 * \return Sparse matrix to subtract from identity matrix.
 */
const arma::sp_mat buildBottomBC();
const arma::sp_mat buildT(); /// Builds sparse matrix with 1s on T only
const arma::sp_mat buildSP(const arma::vec &xn); // Matrix to multiply for source terms of P

/*!
 * \brief Operator to turn a field vector into a scalar vector with the x components.
 * \return Sparse matrix to multiply a field vector by.
 */
const arma::sp_mat field_xProjection();

/*!
 * \brief Operator to turn a field vector into a scalar vector with the z components.
 * \return Sparse matrix to multiply a field vector by.
 */
const arma::sp_mat field_zProjection();

/*!
 * \brief Operator to turn a field vector into a scalar vector with the perpendicular components.
 * \return Sparse matrix to multiply a field vector by.
 */
const arma::sp_mat field_pProjection();

/*!
 * \brief Operator to turn a scalar vector into a field vector with the scalar value for the three components.
 * \return Sparse matrix to multiply a scalar vector by.
 */
const arma::sp_mat field_scalar2field();

/*!
 * \brief Operator to turn a field vector into a scalar vector that is the sum of the field's components (a_x + a_z + a_p)
 * \return Sparse matrix to multiply a field vector by.
 */
const arma::sp_mat field_field2scalar();

/*!
 * \brief Operator to turn a scalar vector into a field vector with the scalar value for the x components.
 * \return Sparse matrix to multiply a scalar vector by.
 */
const arma::sp_mat field_scalar2xfield();

/*!
 * \brief Operator to turn a scalar vector into a field vector with the scalar value for the perp components.
 * \return Sparse matrix to multiply a scalar vector by.
 */
const arma::sp_mat field_scalar2pfield();

/*!
 * \brief Operator to turn a scalar vector into a field vector with the scalar value for the z components.
 * \return Sparse matrix to multiply a scalar vector by.
 */
const arma::sp_mat field_scalar2zfield();

/*!
 * \brief Build an operator to turn a result vector into a scalar vector with ion density values.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildNpscalar();

/*!
 * \brief Build an operator to turn a result vector into a scalar vector with neutral density values.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildNnscalar();

/*!
 * \brief Build an operator to turn a result vector into a scalar vector with ion temperature values.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildTpscalar();

/*!
 * \brief Build an operator to turn a result vector into a scalar vector with neutral temperature values.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildTnscalar();

/*!
 * \brief Build an operator to turn a result vector into a scalar vector with lagrange multiplier values.
 * \return Sparse matrix to multiply a result vector by.
 */
const arma::sp_mat buildGLMscalar();

/*!
 * \brief Operator to find the gradient of a scalar into a field vector.
 * \param leftBC Left boundary condition
 * \param rightBC Right boundary condition
 * \param topBC Top boundary condition
 * \param bottomBC Bottom boundary condition
 * \return Sparse matrix to multiply scalar vector by.
 */
const arma::sp_mat field_scalarGrad(comfi::types::BoundaryCondition leftBC,
                                    comfi::types::BoundaryCondition bottomBC,
                                    comfi::types::BoundaryCondition topBC,
                                    comfi::types::BoundaryCondition rightBC);

/*!
 * \brief Operator to find the divergence of a field vector into a scalar vector.
 * \param LeftBC Left boundary condition
 * \param RightBC Right boundary condition
 * \param UpBC Top boundary condition
 * \param DownBC Bottom boundary condition
 * \return Sparse matrix to multiply field vector by.
 */
const arma::sp_mat field_fieldDiv(comfi::types::BoundaryCondition LeftBC,
                                  comfi::types::BoundaryCondition RightBC,
                                  comfi::types::BoundaryCondition UpBC,
                                  comfi::types::BoundaryCondition DownBC);

/*!
 * \brief Build a sparse matrix with 1s on the magnetic field components to be later subtracted by identity sp matrix.
 * \return Sparse matrix to multiply result vector by.
 */
const arma::sp_mat buildBremove();

/*!
 * \brief Build a sparse matrix with 1s on lagrange multiplier.
 * \return Sparse matrix to multiply result vector by.
 */
const arma::sp_mat buildGLMremove();

/*!
 * \brief Allows dirichlet boundary condition on the bottom boundary for Bz. Fills spots with Bz values to multiply by.
 * \param bg Background data (Bz data required)
 * \return Sparse matrix to multiply a floating value by.
 */
const vcl_sp_mat bottomBz(const comfi::types::BgData &bg);

/*!
 * \brief Allows dirichlet boundary condition on the bottom for Tp.
 * \return Sparse matrix to multiply a floating value of dirichlet condition by.
 */
const vcl_sp_mat bottomTp();

/*!
 * \brief Allows dirichlet boundary condition on the bottom for Tn.
 * \return
 */
const vcl_sp_mat bottomTn();

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
 * \brief saveSolution Saves the solution to the output folder
 * \param x0 Solution matrix.
 * \param ctx Simulation context.
 * \param final If this is the final solution then save under -1.
 * \return Success or fail.
 */
bool saveSolution(const vcl_mat &x0, comfi::types::Context &ctx, const bool final = false);

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
 * \param ctx Simulation context
 * \return initial condition matrix
 */
vcl_mat shock_tube_ic(comfi::types::Context &ctx);

/*!
 * \brief Calculate the initial condition based on Sod's Shock Tube.
 * \param bg Background data, this will be changed in the function.
 * \param op operators
 * \return initial condition vector
 */
std::tuple<arma::vec, const comfi::types::BgData> shock_tube_ic(const comfi::types::Operators &op);

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
 * \param x0 Solution matrix
 * \param ctx Simulation context
 * \return Max characteristic speed in normalized units
 */
double getmaxV(const vcl_mat &x0, comfi::types::Context &ctx);

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
 * \param xn Current time step result matrix
 * \param ctx Simulation context
 * \return Right hand side result solution matrix
 */
vcl_mat computeRHS_Euler(const vcl_mat &xn, comfi::types::Context &ctx);

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
