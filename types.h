/*!
 * \file
 * \brief All data types are found here.
 */
#pragma once

#include <armadillo>
#include "comfi.h"

namespace comfi {

  /*!
 * All the classes can be found here.
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
 * \brief The simulation context. Everything needed is in here.
 */
class Context {
  BgData m_bg;
  const BoundaryCondition m_bc_up = NEUMANN,
                          m_bc_down = NEUMANN,
                          m_bc_right = NEUMANN,
                          m_bc_left = NEUMANN;
  const uint m_flags;
  const arma::uword m_nx, m_nz;
  const bool m_resumed = false;
  const ushort m_num_of_eq = 14;

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
    // Intentionally left blank
  }

  arma::uword nx() { return m_nx; }
  arma::uword nz() { return m_nz; }
  BoundaryCondition bc_up() { return m_bc_up; }
  BoundaryCondition bc_down() { return m_bc_down; }
  BoundaryCondition bc_right() { return m_bc_right; }
  BoundaryCondition bc_left() { return m_bc_left; }
  uint flags() { return m_flags; }
  bool is_resumed() { return m_resumed; }
  arma::uword num_of_eq() { return m_num_of_eq; }
};

} // namespace types
} // namespace mhdsim
