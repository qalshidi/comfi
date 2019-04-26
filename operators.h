/*!
  * \file
  * \brief Operator building
  */
#pragma once

#include "comfi.h"

namespace comfi {

/*!
 * Functions to build the 'operators' (sparse matrices to extract vectors) are found here. The sparse matrices typically reside in \ref mhdsim::types::Operators .
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
} // namespace mhdsim
