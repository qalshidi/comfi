#include <armadillo>
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/forwards.h"
#include "comfi.h"

using namespace viennacl::linalg;
using namespace arma;

sp_mat comfi::routines::computeRi(const vcl_vec &xn_vcl, const comfi::types::Operators &op)
{
  arma::vec xn(num_of_elem);
  viennacl::fast_copy(xn_vcl, xn);

  const uint nnzp = 12;
  umat  Avi = zeros<umat>(nnzp, num_of_grid);
  umat  Avj = zeros<umat>(nnzp, num_of_grid);
  mat   Avv = zeros<mat> (nnzp, num_of_grid);

  const comfi::types::BoundaryCondition Left = op.getLeftBC();
  const comfi::types::BoundaryCondition Right = op.getRightBC();
  const comfi::types::BoundaryCondition Up = op.getUpBC();
  const comfi::types::BoundaryCondition Down = op.getDownBC();

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i and c++ index ii
    const unsigned int  j=(index)/nx; //find j and c++ index jj

    const int           ij=ind(i, j);
    int                 ip1j=ind(i+1, j);
    int                 im1j=ind(i-1, j);
    int                 ijp1=ind(i, j+1);
    int                 ijm1=ind(i, j-1);

    if (i==0 && (Left==comfi::types::MIRROR || Left==comfi::types::NEUMANN)) { im1j = ij; }
    else if (i==0 && (Left==comfi::types::PERIODIC)) { im1j = ind(nx-1, j); }
    else if (i==nx-1 && (Right==comfi::types::MIRROR || Right==comfi::types::NEUMANN)) { ip1j = ij; }
    else if (i==nx-1 && (Right==comfi::types::PERIODIC)) { ip1j=ind(0, j); }
    if (j==0 && Down==comfi::types::NEUMANN) { ijm1 = ij; }
    else if (j==nz-1 && Up==comfi::types::NEUMANN) { ijp1 = ij; }
    if (Left == comfi::types::DIMENSIONLESS) { im1j = ij; }
    if (Right == comfi::types::DIMENSIONLESS) { ip1j = ij; }

    //const double Ns     = 1.0; //Choosing scaling density to be ionosphere (See: Dr. Tu)
    const double Npij   = std::abs(xn(n_p+ij));
    const double Tpij   = std::abs(xn(T_p+ij));
    const double Nnij   = std::abs(xn(n_n+ij));
    const double Tnij   = std::abs(xn(T_n+ij));
    /*
    const double Npiphj = 0.5*(std::abs(xn(n_p+ip1j))+Npij);
    const double Npimhj = 0.5*(std::abs(xn(n_p+im1j))+Npij);
    const double Npijph = 0.5*(std::abs(xn(n_p+ijp1))+Npij);
    const double Nniphj = 0.5*(std::abs(xn(n_n+ip1j))+Nnij);
    const double Nnimhj = 0.5*(std::abs(xn(n_n+im1j))+Nnij);
    const double Nnijph = 0.5*(std::abs(xn(n_n+ijp1))+Nnij);
    const double Tpiphj = 0.5*(std::abs(xn(T_p+ip1j))+Tpij);
    const double Tpimhj = 0.5*(std::abs(xn(T_p+im1j))+Tpij);
    const double Tpijph = 0.5*(std::abs(xn(T_p+ijp1))+Tpij);
    const double Tniphj = 0.5*(std::abs(xn(T_n+ip1j))+Tnij);
    const double Tnimhj = 0.5*(std::abs(xn(T_n+im1j))+Tnij);
    const double Tnijph = 0.5*(std::abs(xn(T_n+ijp1))+Tnij);
    double Nnijmh = 0.5*(std::abs(xn(n_n+ijm1))+Nnij);
    double Npijmh = 0.5*(std::abs(xn(n_p+ijm1))+Npij);
    double Tpijmh = 0.5*(std::abs(xn(T_p+ijm1))+Tpij);
    double Tnijmh = 0.5*(std::abs(xn(T_n+ijm1))+Tnij);
    */
    /*if (j == 0) {
      Nnijmh = 0.5*(Nn0+Nnij);
      Npijmh = 0.5*(Np0+Npij);
      Tpijmh = 0.5*(T0+Tpij);
      Tnijmh = 0.5*(T0+Tnij);
    }*/

    /*
    const double Kpiphj = mhdsim::sol::kappa_p(Tpiphj, Tniphj, Npiphj, Nniphj);
    const double Kpimhj = mhdsim::sol::kappa_p(Tpimhj, Tnimhj, Npimhj, Nnimhj);
    const double Kpijph = mhdsim::sol::kappa_p(Tpijph, Tnijph, Npijph, Nnijph);
    const double Kpijmh = mhdsim::sol::kappa_p(Tpijmh, Tnijmh, Npijmh, Nnijmh);
    const double Kniphj = mhdsim::sol::kappa_n(Tpiphj, Tniphj, Npiphj, Nniphj);
    const double Knimhj = mhdsim::sol::kappa_n(Tpimhj, Tnimhj, Npimhj, Nnimhj);
    const double Knijph = mhdsim::sol::kappa_n(Tpijph, Tnijph, Npijph, Nnijph);
    const double Knijmh = mhdsim::sol::kappa_n(Tpijmh, Tnijmh, Npijmh, Nnijmh);
    */


    //const double nuin   = mhdsim::sol::nu_in(Nnij, 0.5*(Tpij+Tnij));
    //const double nuni   = mhdsim::sol::nu_in(Nnij, 0.5*(Tpij+Tnij))*Npij/Nnij;
    const double nuni   = collisionrate;
    const double nuin   = nuni*Nnij/Npij;

    //const double resij = mhdsim::sol::resistivity(Npij, Nnij, Tpij, Tnij);

    //const double irate = mhdsim::sol::ionization_coeff(Tnij);
    //const double rrate = mhdsim::sol::recomb_coeff(Tpij);

    int p=-1;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    // density
    //p=p+1; vi(p)=n_p+ij; vj(p)=n_p+ij;
    //vv(p) = rrate;
    //p=p+1; vi(p)=n_p+ij; vj(p)=n_n+ij;
    //vv(p) = -irate;
    //p=p+1; vi(p)=n_n+ij; vj(p)=n_p+ij;
    //vv(p) = -rrate;
    //p=p+1; vi(p)=n_n+ij; vj(p)=n_n+ij;
    //vv(p) = rrate;
    //p=p+1; vi(p)=n_p+ij; vj(p)=n_n+ij;
    //vv(p) = -irate;
    //p=p+1; vi(p)=n_n+ij; vj(p)=n_p+ij;
    //vv(p) = -rrate;
    //p=p+1; vi(p)=n_n+ij; vj(p)=n_n+ij;
    //vv(p) = irate;
    // ion speed
    p=p+1; vi(p)=Vx+ij; vj(p)=Vx+ij;
    vv(p) = nuin;// - irate;
    p=p+1; vi(p)=Vx+ij; vj(p)=Ux+ij;
    vv(p) = -nuni;// + rrate;
    p=p+1; vi(p)=Vz+ij; vj(p)=Vz+ij;
    vv(p) = nuin;// - irate;
    p=p+1; vi(p)=Vz+ij; vj(p)=Uz+ij;
    vv(p) = -nuni;// + rrate;
    p=p+1; vi(p)=Vp+ij; vj(p)=Vp+ij;
    vv(p) = nuin;// - irate;
    p=p+1; vi(p)=Vp+ij; vj(p)=Up+ij;
    vv(p) = -nuni;// + rrate;
    // neutral speed
    p=p+1; vi(p)=Ux+ij; vj(p)=Ux+ij;
    vv(p) = nuni;// - rrate;
    p=p+1; vi(p)=Ux+ij; vj(p)=Vx+ij;
    vv(p) = -nuin;// + irate;
    p=p+1; vi(p)=Uz+ij; vj(p)=Uz+ij;
    vv(p) = nuni;// - rrate;
    p=p+1; vi(p)=Uz+ij; vj(p)=Vz+ij;
    vv(p) = -nuin;// + irate;
    p=p+1; vi(p)=Up+ij; vj(p)=Up+ij;
    vv(p) = nuni;// - rrate;
    p=p+1; vi(p)=Up+ij; vj(p)=Vp+ij;
    vv(p) = -nuin;// + irate;
    // Pressure
    /*
    p=p+1; vi(p)=T_p+ij; vj(p)=T_p+ij;
    vv(p) = +nuin + two_thirds*((Kpiphj + Kpimhj)/dx2 + (Kpijph + Kpijmh)/dz2)/sNpij;
    p=p+1; vi(p)=T_p+ij; vj(p)=T_n+ij;
    vv(p) = -nuin;
    p=p+1; vi(p)=T_p+ij; vj(p)=T_p+ip1j;
    vv(p) = -two_thirds*(Kpiphj/sNpij)/dx2;
    p=p+1; vi(p)=T_p+ij; vj(p)=T_p+im1j;
    vv(p) = -two_thirds*(Kpimhj/sNpij)/dx2;
    p=p+1; vi(p)=T_p+ij; vj(p)=T_p+ijp1;
    vv(p) = -two_thirds*(Kpijph/sNpij)/dz2;
    if (j != 0) {
    p=p+1; vi(p)=T_p+ij; vj(p)=T_p+ijm1;
    vv(p) = -two_thirds*(Kpijmh/sNpij)/dz2;
    }
    p=p+1; vi(p)=T_n+ij; vj(p)=T_n+ij;
    vv(p) = +nuni + two_thirds*((Kniphj + Knimhj)/dx2 + (Knijph + Knijmh)/dz2)/sNnij;
    p=p+1; vi(p)=T_n+ij; vj(p)=T_p+ij;
    vv(p) = -nuni;
    p=p+1; vi(p)=T_n+ij; vj(p)=T_n+ip1j;
    vv(p) = -two_thirds*(Kniphj/sNnij)/dx2;
    p=p+1; vi(p)=T_n+ij; vj(p)=T_n+im1j;
    vv(p) = -two_thirds*(Knimhj/sNnij)/dx2;
    p=p+1; vi(p)=T_n+ij; vj(p)=T_n+ijp1;
    vv(p) = -two_thirds*(Knijph/sNnij)/dz2;
    if (j != 0) {
    p=p+1; vi(p)=T_n+ij; vj(p)=T_n+ijm1;
    vv(p) = -two_thirds*(Knijmh/sNnij)/dz2;
    }
    */
    // Magnetic field
//    p=p+1; vi(p)=Bx+ij; vj(p)=Bx+ij;
//    vv(p) = + 2.0*resij/dx2 + 2.0*resij/dz2;
//    p=p+1; vi(p)=Bx+ij; vj(p)=Bx+ip1j;
//    if (i==nx-1 && Right==mhdsim::types::MIRROR) { vv(p) = resij/dx2; } else { vv(p) = -resij/dx2; }
//    p=p+1; vi(p)=Bx+ij; vj(p)=Bx+im1j;
//    if (i==0 && Left==mhdsim::types::MIRROR) { vv(p) = resij/dx2; } else { vv(p) = -resij/dx2; }
//    p=p+1; vi(p)=Bx+ij; vj(p)=Bx+ijp1;
//    vv(p) = -resij/dz2;
//    //if (j != 0) {
//    p=p+1; vi(p)=Bx+ij; vj(p)=Bx+ijm1;
//    vv(p) = -resij/dz2;
//    //}
//    p=p+1; vi(p)=Bz+ij; vj(p)=Bz+ij;
//    //vv(p) = + 2.0*resij/dx2 + 2.0*resij/dz2;
//    p=p+1; vi(p)=Bz+ij; vj(p)=Bz+ip1j;
//    vv(p) = -resij/dx2;
//    p=p+1; vi(p)=Bz+ij; vj(p)=Bz+im1j;
//    vv(p) = -resij/dx2;
//    p=p+1; vi(p)=Bz+ij; vj(p)=Bz+ijp1;
//    vv(p) = -resij/dz2;
    //if (j != 0) {
    //p=p+1; vi(p)=Bz+ij; vj(p)=Bz+ijm1;
    //vv(p) = -resij/dz2;
    //}
    //p=p+1; vi(p)=Bp+ij; vj(p)=Bp+ij;
    //vv(p) = + 2.0*resij/dx2 + 2.0*resij/dz2;
    //p=p+1; vi(p)=Bp+ij; vj(p)=Bp+ip1j;
    //vv(p) = -resij/dx2;
    //p=p+1; vi(p)=Bp+ij; vj(p)=Bp+im1j;
    //vv(p) = -resij/dx2;
    //p=p+1; vi(p)=Bp+ij; vj(p)=Bp+ijp1;
    //vv(p) = -resij/dz2;
    //p=p+1; vi(p)=Bp+ij; vj(p)=Bp+ijm1;
    //vv(p) = -resij/dz2;

    // collect
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

vcl_mat comfi::routines::Re_MUSCL(const vcl_mat &xn, const double t, comfi::types::Context &ctx)
{
  const vcl_mat xn_ip1 = comfi::operators::ip1(xn, ctx);
  const vcl_mat xn_im1 = comfi::operators::im1(xn, ctx);
  const vcl_mat xn_jp1 = comfi::operators::jp1(xn, ctx);
  const vcl_mat xn_jm1 = comfi::operators::jm1(xn, ctx);

  const vcl_mat dxn_iph = xn_ip1-xn;
  const vcl_mat dxn_imh = xn-xn_im1;
  const vcl_mat dxn_jph = xn_jp1-xn;
  const vcl_mat dxn_jmh = xn-xn_jm1;

  static const vcl_mat eps = viennacl::scalar_matrix<double>(xn.size1(), xn.size2(), 1.e-100);

  const vcl_mat r_i   = element_div(dxn_imh, dxn_iph+eps);
  const vcl_mat r_ip1 = element_div(dxn_iph, comfi::operators::ip1(xn_ip1, ctx)-xn_ip1+eps);
  const vcl_mat r_im1 = element_div(xn_im1-comfi::operators::im1(xn_im1, ctx), dxn_imh+eps);
  const vcl_mat r_j   = element_div(dxn_jmh, dxn_jph+eps);
  const vcl_mat r_jp1 = element_div(dxn_jph, comfi::operators::jp1(xn_jp1, ctx)-xn_jp1+eps);
  const vcl_mat r_jm1 = element_div(xn_jm1-comfi::operators::jm1(xn_jm1, ctx), dxn_jmh+eps);

  //extrapolated cell edge variables
  vcl_mat Lxn_iph = xn     + 0.5*element_prod(comfi::routines::fluxl(r_i), dxn_imh);
  vcl_mat Lxn_imh = xn_im1 - 0.5*element_prod(comfi::routines::fluxl(r_im1), dxn_imh);
  vcl_mat Rxn_iph = xn_ip1 + 0.5*element_prod(comfi::routines::fluxl(r_ip1), dxn_iph);
  vcl_mat Rxn_imh = xn     - 0.5*element_prod(comfi::routines::fluxl(r_i), dxn_iph);
  vcl_mat Lxn_jph = xn     + 0.5*element_prod(comfi::routines::fluxl(r_j), dxn_jmh);
  vcl_mat Lxn_jmh = xn_jm1 - 0.5*element_prod(comfi::routines::fluxl(r_jm1), dxn_jmh);
  vcl_mat Rxn_jph = xn_jp1 + 0.5*element_prod(comfi::routines::fluxl(r_jp1), dxn_jph);
  vcl_mat Rxn_jmh = xn     - 0.5*element_prod(comfi::routines::fluxl(r_j), dxn_jph);

  // BOUNDARY CONDITIONS
  //mhdsim::routines::bottomBC(Lxn_jmh,Rxn_jmh,t,op,bg);
  //mhdsim::routines::bottombc_shock_tube(Lxn_jmh, Rxn_jmh, op);
  //mhdsim::routines::topbc_shock_tube(Lxn_jph, Rxn_jph, op);
  //comfi::routines::topbc_soler(Lxn_jph, Rxn_jph, op);
  //mhdsim::routines::topbc_driver(Lxn_jph, Rxn_jph, t, op);
  //comfi::routines::bottombc_soler(Lxn_jmh, Rxn_jmh, op);

  /*
  // Fast mode speed eigenvalues
  vcl_vec Leig_iph = comfi::routines::fast_speed_x(Lxn_iph, op);
  vcl_vec Reig_iph = comfi::routines::fast_speed_x(Rxn_iph, op);
  vcl_vec Leig_jph = comfi::routines::fast_speed_z(Lxn_jph, op);
  vcl_vec Reig_jph = comfi::routines::fast_speed_z(Rxn_jph, op);
  vcl_vec Leig_imh = comfi::routines::fast_speed_x(Lxn_imh, op);
  vcl_vec Reig_imh = comfi::routines::fast_speed_x(Rxn_imh, op);
  vcl_vec Leig_jmh = comfi::routines::fast_speed_z(Lxn_jmh, op);
  vcl_vec Reig_jmh = comfi::routines::fast_speed_z(Rxn_jmh, op);
  viennacl::ocl::program & eig_prog  = viennacl::ocl::current_context().get_program("largest_eig");
  viennacl::ocl::kernel  & eig_kernel = eig_prog.get_kernel("largest_eig");
  vcl_vec a_imh(num_of_elem); viennacl::ocl::enqueue(eig_kernel(Leig_imh, Reig_imh, a_imh, cl_uint(Leig_imh.size())));
  vcl_vec a_iph(num_of_elem); viennacl::ocl::enqueue(eig_kernel(Leig_iph, Reig_iph, a_iph, cl_uint(Leig_iph.size())));
  vcl_vec a_jmh(num_of_elem); viennacl::ocl::enqueue(eig_kernel(Leig_jmh, Reig_jmh, a_jmh, cl_uint(Leig_jmh.size())));
  vcl_vec a_jph(num_of_elem); viennacl::ocl::enqueue(eig_kernel(Leig_jph, Reig_jph, a_jph, cl_uint(Leig_jph.size())));

  // Local velocity eigenvalues
  Leig_iph = element_fabs(element_div(prod(op.PEVx, Lxn_iph), prod(op.PN, Lxn_iph)));
  Reig_iph = element_fabs(element_div(prod(op.PEVx, Rxn_iph), prod(op.PN, Rxn_iph)));
  Leig_jph = element_fabs(element_div(prod(op.PEVz, Lxn_jph), prod(op.PN, Lxn_jph)));
  Reig_jph = element_fabs(element_div(prod(op.PEVz, Rxn_jph), prod(op.PN, Rxn_jph)));
  Leig_imh = element_fabs(element_div(prod(op.PEVx, Lxn_imh), prod(op.PN, Lxn_imh)));
  Reig_imh = element_fabs(element_div(prod(op.PEVx, Rxn_imh), prod(op.PN, Rxn_imh)));
  Leig_jmh = element_fabs(element_div(prod(op.PEVz, Lxn_jmh), prod(op.PN, Lxn_jmh)));
  Reig_jmh = element_fabs(element_div(prod(op.PEVz, Rxn_jmh), prod(op.PN, Rxn_jmh)));
  viennacl::ocl::enqueue(eig_kernel(Leig_imh, a_imh, a_imh, cl_uint(Leig_imh.size())));
  viennacl::ocl::enqueue(eig_kernel(Reig_imh, a_imh, a_imh, cl_uint(Reig_imh.size())));
  viennacl::ocl::enqueue(eig_kernel(Leig_iph, a_iph, a_iph, cl_uint(Leig_iph.size())));
  viennacl::ocl::enqueue(eig_kernel(Reig_iph, a_iph, a_iph, cl_uint(Reig_iph.size())));
  viennacl::ocl::enqueue(eig_kernel(Leig_jph, a_jph, a_jph, cl_uint(Leig_jph.size())));
  viennacl::ocl::enqueue(eig_kernel(Reig_jph, a_jph, a_jph, cl_uint(Reig_jph.size())));
  viennacl::ocl::enqueue(eig_kernel(Leig_jmh, a_jmh, a_jmh, cl_uint(Leig_jmh.size())));
  viennacl::ocl::enqueue(eig_kernel(Reig_jmh, a_jmh, a_jmh, cl_uint(Leig_jmh.size())));
  */



  // SCALARS

  // LAX-FRIEDRICHS FLUX
  /*
  const vcl_vec Fximh = 0.5*(mhdsim::routines::Fx(Lxn_imh, sNp, op)+mhdsim::routines::Fx(Rxn_imh, sNp, op))
                      - element_prod(a_imh,(Rxn_imh - Lxn_imh));
  const vcl_vec Fxiph = 0.5*(mhdsim::routines::Fx(Lxn_iph, sNp, op)+mhdsim::routines::Fx(Rxn_iph, sNp, op))
                      - element_prod(a_iph,(Rxn_iph-Lxn_iph));
  */
  /*
  const vcl_vec Fzjmh = 0.5*(comfi::routines::Fz(Lxn_jmh, Np, op)+comfi::routines::Fz(Rxn_jmh, Np, op))
                      - element_prod(a_jmh,(Rxn_jmh-Lxn_jmh));
  const vcl_vec Fzjph = 0.5*(comfi::routines::Fz(Lxn_jph, Np, op)+comfi::routines::Fz(Rxn_jph, Np, op))
                      - element_prod(a_jph,(Rxn_jph-Lxn_jph));
*/
  return xn; //-1.0*(Fxiph-Fximh)/dx
         //-1.0*(Fzjph-Fzjmh)/dz
         //+ prod(op.f2V, v_collission_source)
         //- prod(op.f2U, v_collission_source)
         //+ prod(op.f2U, u_collission_source)
         //- prod(op.f2V, u_collission_source);
         //+ prod(op.f2V, me_nu_J)
         //+ prod(op.f2V, vsource)
         //- prod(op.f2U, me_nu_J)
         //- prod(op.f2U, vsource)
         //- prod(op.SG, xn)
         //+ boundaryconditions
         //+ prod(op.s2Np, isource)
         //- prod(op.s2Nn, isource)
         //+ prod(op.s2Tp, TpdivV)/3.0;
         //+ prod(op.s2Tn, TndivU)/3.0;
         //+ prod(op.s2Tp, nuin_dV2)/3.0
         //+ prod(op.s2Tn, nuni_dV2)/3.0
         //- two_thirds*prod(op.s2Tn, L)
         //+ prod(op.f2B, gNpxJxBoverN2) // Hall term source term
         //- prod(op.f2B,gradrescrossJ);
}

vcl_vec comfi::routines::Re_MUSCL(const vcl_vec &xn, const double t, const comfi::types::Operators &op, const comfi::types::BgData &bg)
{
  const vcl_vec xn_ip1 = prod(op.Pip1,xn);
  const vcl_vec xn_ip2 = prod(op.Pip2,xn);
  const vcl_vec xn_im1 = prod(op.Pim1,xn);
  const vcl_vec xn_im2 = prod(op.Pim2,xn);
  const vcl_vec xn_jp1 = prod(op.Pjp1,xn);
  const vcl_vec xn_jp2 = prod(op.Pjp2,xn);
  const vcl_vec xn_jm1 = prod(op.Pjm1,xn);
  const vcl_vec xn_jm2 = prod(op.Pjm2,xn);

  const vcl_vec dxn_iph = xn_ip1-xn;
  const vcl_vec dxn_imh = xn-xn_im1;
  const vcl_vec dxn_jph = xn_jp1-xn;
  const vcl_vec dxn_jmh = xn-xn_jm1;

  const vcl_vec r_i   = element_div(dxn_imh,dxn_iph);
  const vcl_vec r_ip1 = element_div(dxn_iph,(xn_ip2-xn_ip1));
  const vcl_vec r_im1 = element_div((xn_im1-xn_im2),dxn_imh);
  const vcl_vec r_j   = element_div(dxn_jmh,dxn_jph);
  const vcl_vec r_jp1 = element_div(dxn_jph,(xn_jp2-xn_jp1));
  const vcl_vec r_jm1 = element_div((xn_jm1-xn_jm2),dxn_jmh);

  viennacl::ocl::program & phi_prog  = viennacl::ocl::current_context().get_program("fluxl");
  viennacl::ocl::kernel  & my_kernel = phi_prog.get_kernel("fluxl");

  vcl_vec phi_i(num_of_elem);   viennacl::ocl::enqueue(my_kernel(r_i, phi_i, cl_uint(r_i.size())));
  vcl_vec phi_ip1(num_of_elem); viennacl::ocl::enqueue(my_kernel(r_ip1, phi_ip1, cl_uint(r_ip1.size())));
  vcl_vec phi_im1(num_of_elem); viennacl::ocl::enqueue(my_kernel(r_im1, phi_im1, cl_uint(r_im1.size())));
  vcl_vec phi_j(num_of_elem);   viennacl::ocl::enqueue(my_kernel(r_j, phi_j, cl_uint(r_j.size())));
  vcl_vec phi_jm1(num_of_elem); viennacl::ocl::enqueue(my_kernel(r_jm1, phi_jm1, cl_uint(r_jm1.size())));
  vcl_vec phi_jp1(num_of_elem); viennacl::ocl::enqueue(my_kernel(r_jp1, phi_jp1, cl_uint(r_jp1.size())));

  //extrapolated cell edge variables
  vcl_vec Lxn_iph = xn     + 0.5*element_prod(phi_i,dxn_imh);
  vcl_vec Lxn_imh = xn_im1 - 0.5*element_prod(phi_im1,dxn_imh);
  vcl_vec Rxn_iph = xn_ip1 + 0.5*element_prod(phi_ip1,dxn_iph);
  vcl_vec Rxn_imh = xn     - 0.5*element_prod(phi_i,dxn_iph);
  vcl_vec Lxn_jph = xn     + 0.5*element_prod(phi_j,dxn_jmh);
  vcl_vec Lxn_jmh = xn_jm1 - 0.5*element_prod(phi_jm1,dxn_jmh);
  vcl_vec Rxn_jph = xn_jp1 + 0.5*element_prod(phi_jp1,dxn_jph);
  vcl_vec Rxn_jmh = xn     - 0.5*element_prod(phi_j,dxn_jph);

  // BOUNDARY CONDITIONS
  //mhdsim::routines::bottomBC(Lxn_jmh,Rxn_jmh,t,op,bg);
  //mhdsim::routines::bottombc_shock_tube(Lxn_jmh, Rxn_jmh, op);
  //mhdsim::routines::topbc_shock_tube(Lxn_jph, Rxn_jph, op);
  comfi::routines::topbc_soler(Lxn_jph, Rxn_jph, op);
  //mhdsim::routines::topbc_driver(Lxn_jph, Rxn_jph, t, op);
  comfi::routines::bottombc_soler(Lxn_jmh, Rxn_jmh, op);

  // Fast mode speed eigenvalues
  vcl_vec Leig_iph = comfi::routines::fast_speed_x(Lxn_iph, op);
  vcl_vec Reig_iph = comfi::routines::fast_speed_x(Rxn_iph, op);
  vcl_vec Leig_jph = comfi::routines::fast_speed_z(Lxn_jph, op);
  vcl_vec Reig_jph = comfi::routines::fast_speed_z(Rxn_jph, op);
  vcl_vec Leig_imh = comfi::routines::fast_speed_x(Lxn_imh, op);
  vcl_vec Reig_imh = comfi::routines::fast_speed_x(Rxn_imh, op);
  vcl_vec Leig_jmh = comfi::routines::fast_speed_z(Lxn_jmh, op);
  vcl_vec Reig_jmh = comfi::routines::fast_speed_z(Rxn_jmh, op);
  viennacl::ocl::program & eig_prog  = viennacl::ocl::current_context().get_program("largest_eig");
  viennacl::ocl::kernel  & eig_kernel = eig_prog.get_kernel("largest_eig");
  vcl_vec a_imh(num_of_elem); viennacl::ocl::enqueue(eig_kernel(Leig_imh, Reig_imh, a_imh, cl_uint(Leig_imh.size())));
  vcl_vec a_iph(num_of_elem); viennacl::ocl::enqueue(eig_kernel(Leig_iph, Reig_iph, a_iph, cl_uint(Leig_iph.size())));
  vcl_vec a_jmh(num_of_elem); viennacl::ocl::enqueue(eig_kernel(Leig_jmh, Reig_jmh, a_jmh, cl_uint(Leig_jmh.size())));
  vcl_vec a_jph(num_of_elem); viennacl::ocl::enqueue(eig_kernel(Leig_jph, Reig_jph, a_jph, cl_uint(Leig_jph.size())));

  // Local velocity eigenvalues
  Leig_iph = element_fabs(element_div(prod(op.PEVx, Lxn_iph), prod(op.PN, Lxn_iph)));
  Reig_iph = element_fabs(element_div(prod(op.PEVx, Rxn_iph), prod(op.PN, Rxn_iph)));
  Leig_jph = element_fabs(element_div(prod(op.PEVz, Lxn_jph), prod(op.PN, Lxn_jph)));
  Reig_jph = element_fabs(element_div(prod(op.PEVz, Rxn_jph), prod(op.PN, Rxn_jph)));
  Leig_imh = element_fabs(element_div(prod(op.PEVx, Lxn_imh), prod(op.PN, Lxn_imh)));
  Reig_imh = element_fabs(element_div(prod(op.PEVx, Rxn_imh), prod(op.PN, Rxn_imh)));
  Leig_jmh = element_fabs(element_div(prod(op.PEVz, Lxn_jmh), prod(op.PN, Lxn_jmh)));
  Reig_jmh = element_fabs(element_div(prod(op.PEVz, Rxn_jmh), prod(op.PN, Rxn_jmh)));
  viennacl::ocl::enqueue(eig_kernel(Leig_imh, a_imh, a_imh, cl_uint(Leig_imh.size())));
  viennacl::ocl::enqueue(eig_kernel(Reig_imh, a_imh, a_imh, cl_uint(Reig_imh.size())));
  viennacl::ocl::enqueue(eig_kernel(Leig_iph, a_iph, a_iph, cl_uint(Leig_iph.size())));
  viennacl::ocl::enqueue(eig_kernel(Reig_iph, a_iph, a_iph, cl_uint(Reig_iph.size())));
  viennacl::ocl::enqueue(eig_kernel(Leig_jph, a_jph, a_jph, cl_uint(Leig_jph.size())));
  viennacl::ocl::enqueue(eig_kernel(Reig_jph, a_jph, a_jph, cl_uint(Reig_jph.size())));
  viennacl::ocl::enqueue(eig_kernel(Leig_jmh, a_jmh, a_jmh, cl_uint(Leig_jmh.size())));
  viennacl::ocl::enqueue(eig_kernel(Reig_jmh, a_jmh, a_jmh, cl_uint(Leig_jmh.size())));


  const vcl_vec xn_iph = 0.5*(Lxn_iph+Rxn_iph);
  const vcl_vec xn_jph = 0.5*(Lxn_jph+Rxn_jph);
  const vcl_vec xn_imh = 0.5*(Lxn_imh+Rxn_imh);
  const vcl_vec xn_jmh = 0.5*(Lxn_jmh+Rxn_jmh);

  // SCALARS
  const vcl_vec Np = prod(op.Nps,xn);
  const vcl_vec Nn = prod(op.Nns,xn);
  const vcl_vec Tp = element_fabs(prod(op.Tps,xn));
  const vcl_vec Tn = element_fabs(prod(op.Tns,xn));
  const vcl_vec Npiph = prod(op.Nps,xn_iph);
  const vcl_vec Nniph = prod(op.Nns,xn_iph);
  const vcl_vec Tpiph = element_fabs(prod(op.Tps,xn_iph));
  const vcl_vec Tniph = element_fabs(prod(op.Tns,xn_iph));
  const vcl_vec Npimh = prod(op.Nps,xn_imh);
  const vcl_vec Nnimh = prod(op.Nns,xn_imh);
  const vcl_vec Tpimh = element_fabs(prod(op.Tps,xn_imh));
  const vcl_vec Tnimh = element_fabs(prod(op.Tns,xn_imh));
  const vcl_vec Npjph = prod(op.Nps,xn_jph);
  const vcl_vec Nnjph = prod(op.Nns,xn_jph);
  const vcl_vec Tpjph = element_fabs(prod(op.Tps,xn_jph));
  const vcl_vec Tnjph = element_fabs(prod(op.Tns,xn_jph));
  const vcl_vec Npjmh = prod(op.Nps,xn_jmh);
  const vcl_vec Nnjmh = prod(op.Nns,xn_jmh);
  const vcl_vec Tpjmh = element_fabs(prod(op.Tps,xn_jmh));
  const vcl_vec Tnjmh = element_fabs(prod(op.Tns,xn_jmh));

  /*
  // FIELDS
  //const vcl_vec NVf = prod(op.Vf,xn);
  const vcl_vec Vf = element_div(prod(op.Vf,xn), element_fabs(prod(op.s2f,Np)));
  //const vcl_vec NUf = prod(op.Uf,xn);
  const vcl_vec Uf = element_div(prod(op.Uf,xn), element_fabs(prod(op.s2f,Nn)));
  const vcl_vec Bf = prod(op.Bf, xn);
  const vcl_vec Biph = prod(op.Bf,xn_iph);
  const vcl_vec Bjph = prod(op.Bf,xn_jph);
  const vcl_vec Bimh = prod(op.Bf,xn_imh);
  const vcl_vec Bjmh = prod(op.Bf,xn_jmh);
  const vcl_vec Jf = curl_tvd(Biph, Bimh, Bjph, Bjmh, op);
  //const vcl_vec Jf = prod(op.curl,Bf);
  */
  /*
  // SCALARS DERIVED
  const vcl_vec res_iph = mhdsim::sol::resistivity(Npiph, Nniph, Tpiph, Tniph);
  const vcl_vec res_imh = mhdsim::sol::resistivity(Npimh, Nnimh, Tpimh, Tnimh);
  const vcl_vec res_jph = mhdsim::sol::resistivity(Npjph, Nnjph, Tpjph, Tnjph);
  const vcl_vec res_jmh = mhdsim::sol::resistivity(Npjmh, Nnjmh, Tpjmh, Tnjmh);
  const vcl_vec gradres = mhdsim::routines::grad_tvd(res_iph, res_imh, res_jph, res_jmh, op);
  // B SOURCE
  const vcl_vec gradrescrossJ = mhdsim::routines::cross_prod(gradres, Jf, op);
  const vcl_vec gradNp = mhdsim::routines::grad_tvd(Npiph, Npimh, Npjph, Npjmh, op);
  vcl_vec gNpxJxBoverN2 = mhdsim::routines::cross_prod(gradNp, Jf, op);
  gNpxJxBoverN2 = mhdsim::routines::cross_prod(gNpxJxBoverN2, Bf, op);
  gNpxJxBoverN2 = element_div(gNpxJxBoverN2, q*prod(op.s2f, Np));
  gNpxJxBoverN2 = element_div(gNpxJxBoverN2, prod(op.s2f, Np));
*/

  // V source
  //const vcl_vec nuin = mhdsim::sol::nu_in(element_fabs(Nn), 0.5*(Tp+Tn));
  //const vcl_vec nuni = element_div(element_prod(nuin, Np), Nn);
  const vcl_vec nuni = viennacl::scalar_vector<double>(num_of_grid, collisionrate);
  const vcl_vec nuin = element_div(element_prod(nuni, Nn), Np);
  const vcl_vec NUf = prod(op.Uf, xn);
  const vcl_vec NVf = prod(op.Vf, xn);
  const vcl_vec v_collission_source = element_prod(prod(op.s2f, nuni), NUf);
  const vcl_vec u_collission_source = element_prod(prod(op.s2f, nuin), NVf);

  // E SOURCE
  /*
  const vcl_vec nuen = mhdsim::sol::nu_en(element_fabs(Nn), 0.5*(Tp+Tn));
  const vcl_vec dV2  = mhdsim::routines::dot_prod(Vf-Uf, Vf-Uf, op);
  const vcl_vec nuin_dV2 = element_prod(nuin, dV2);
  const vcl_vec nuni_dV2 = element_prod(element_div(element_fabs(Np), element_fabs(Nn)), nuin_dV2);
  // necessary for ion+electron fluid
  const vcl_vec nuen_nuin = nuen-nuin;
  const vcl_vec me_nu_J = m_e/q*element_prod(prod(op.s2f, nuen_nuin), Jf);
  */
  //const vcl_vec divV = prod(op.div, Vf);
  //const vcl_vec divU = prod(op.div, Uf);
  /*
  vcl_vec Vf_iph = prod(op.Vf, xn_iph);
  vcl_vec Vx_iph = prod(op.fdotx, Vf_iph);
  Vx_iph = element_div(Vx_iph, prod(op.Nps, xn_iph));
  vcl_vec Vf_imh = prod(op.Vf, xn_imh);
  vcl_vec Vx_imh = prod(op.fdotx, Vf_imh);
  Vx_imh = element_div(Vx_imh, prod(op.Nps, xn_imh));
  vcl_vec Vf_jph = prod(op.Vf, xn_jph);
  vcl_vec Vz_jph = prod(op.fdotz, Vf_jph);
  Vz_jph = element_div(Vz_jph, prod(op.Nps, xn_jph));
  vcl_vec Vf_jmh = prod(op.Vf, xn_jmh);
  vcl_vec Vz_jmh = prod(op.fdotz, Vf_jmh);
  Vz_jmh = element_div(Vz_jmh, prod(op.Nps, xn_jmh));
  const vcl_vec divV = mhdsim::routines::div_tvd(Vx_iph, Vx_imh, Vz_jph, Vz_jmh);
  const vcl_vec TpdivV = element_prod(Tp, divV);
  vcl_vec Uf_iph = prod(op.Uf, xn_iph);
  vcl_vec Ux_iph = prod(op.fdotx, Uf_iph);
  Ux_iph = element_div(Ux_iph, prod(op.Nns, xn_iph));
  vcl_vec Uf_imh = prod(op.Uf, xn_imh);
  vcl_vec Ux_imh = prod(op.fdotx, Uf_imh);
  Ux_imh = element_div(Ux_imh, prod(op.Nns, xn_imh));
  vcl_vec Uf_jph = prod(op.Uf, xn_jph);
  vcl_vec Uz_jph = prod(op.fdotz, Uf_jph);
  Uz_jph = element_div(Uz_jph, prod(op.Nns, xn_jph));
  vcl_vec Uf_jmh = prod(op.Uf, xn_jmh);
  vcl_vec Uz_jmh = prod(op.fdotz, Uf_jmh);
  Uz_jmh = element_div(Uz_jmh, prod(op.Nns, xn_jmh));
  const vcl_vec divU = mhdsim::routines::div_tvd(Ux_iph, Ux_imh, Uz_jph, Uz_jmh);
  const vcl_vec TndivU = element_prod(Tn, divU);
  */

  /*
  static arma::vec plambda;
  static bool plambda_isloaded = false;
  if (!plambda_isloaded)
  {
    plambda_isloaded = plambda.load("input/plambda.csv");
  }
  vcl_vec lambda = routines::polyval(plambda, element_log10(Tn*T_0));
  const static vcl_vec ten = viennacl::scalar_vector<double>(lambda.size(), 10.0);
  lambda = 1e-7/(arma::datum::k*T_0) * t_0 * (n_0*1e-6) * viennacl::linalg::element_pow(ten ,lambda);
  vcl_vec alphan2overnn = element_prod(Np, Np+Nn);
  alphan2overnn = element_div(alphan2overnn, Nn);
  const vcl_vec L = element_prod(lambda, alphan2overnn);
  */

  /*
  const static vcl_vec one_hundred8 = viennacl::scalar_vector<double>(Tn.size(), 100.8);
  vcl_vec lambda = 19.54 * element_log10(Tn*T_0) - one_hundred8;
  const static vcl_vec ten = viennacl::scalar_vector<double>(lambda.size(), 10);
  lambda = 1e-7/(arma::datum::k*T_0) * t_0 * (n_0*1e-6) * element_pow(ten ,lambda);
  const vcl_vec L = element_prod(lambda, Np);
  static int timestep = 0;
  timestep++;
  util::saveScalar(lambda, "lambda", timestep);
  util::saveScalar(L, "L", timestep);
  */

  //Chemistry sources

  /*
  vcl_vec ionization = element_prod(Np, mhdsim::sol::ionization_coeff(Tp));
  ionization = element_prod(Nn, ionization);
  vcl_vec recombination = element_prod(Np, mhdsim::sol::recomb_coeff(Tp));
  recombination = element_prod(Np, recombination);
  const vcl_vec isource = ionization-recombination;
  const vcl_vec vsource = element_prod(prod(op.s2f, ionization), Vf) - element_prod(prod(op.s2f, recombination), Uf);
  */

  //Bottom BC

  // Sources due to implicit terms dirichlet bottom conditions
  /*
  static const double Ns     = 1.0; //Choosing scaling density to be ionosphere (See: Dr. Tu)
  static const vcl_vec bottomTpsource = viennacl::scalar_vector<double>(num_of_elem, T0*two_thirds*mhdsim::sol::kappa_p(T0, T0, Np0, Nn0)/Np0/dz2);
  static const vcl_vec bottomTnsource = viennacl::scalar_vector<double>(num_of_elem, T0*two_thirds*mhdsim::sol::kappa_n(T0, T0, Np0, Nn0)/Nn0/dz2);
  static const vcl_vec bottomBzsource = viennacl::scalar_vector<double>(num_of_elem, mhdsim::sol::resistivity(Np0, Nn0, T0, T0)/dz2);
  static const vcl_sp_mat bottomTp = mhdsim::operators::bottomTp();
  static const vcl_sp_mat bottomTn = mhdsim::operators::bottomTn();
  static const vcl_sp_mat bottomBz = mhdsim::operators::bottomBz(bg);
  static const vcl_vec boundaryconditions = prod(bottomBz, bottomBzsource) + prod(bottomTp, bottomTpsource) + prod(bottomTn, bottomTnsource);
  */

  //vcl_vec sNp = viennacl::scalar_vector<double>(num_of_grid, Ns);
  //sNp = element_div(element_div(sNp, Np), Np);
  //static const vcl_vec one = viennacl::scalar_vector<double>(num_of_grid, 1.0);
  //sNp = one + sNp;
  //sNp = element_prod(Np, sNp);

  // LAX-FRIEDRICHS FLUX
  /*
  const vcl_vec Fximh = 0.5*(mhdsim::routines::Fx(Lxn_imh, sNp, op)+mhdsim::routines::Fx(Rxn_imh, sNp, op))
                      - element_prod(a_imh,(Rxn_imh - Lxn_imh));
  const vcl_vec Fxiph = 0.5*(mhdsim::routines::Fx(Lxn_iph, sNp, op)+mhdsim::routines::Fx(Rxn_iph, sNp, op))
                      - element_prod(a_iph,(Rxn_iph-Lxn_iph));
  */
  const vcl_vec Fzjmh = 0.5*(comfi::routines::Fz(Lxn_jmh, Np, op)+comfi::routines::Fz(Rxn_jmh, Np, op))
                      - element_prod(a_jmh,(Rxn_jmh-Lxn_jmh));
  const vcl_vec Fzjph = 0.5*(comfi::routines::Fz(Lxn_jph, Np, op)+comfi::routines::Fz(Rxn_jph, Np, op))
                      - element_prod(a_jph,(Rxn_jph-Lxn_jph));

  return //-1.0*(Fxiph-Fximh)/dx
         -1.0*(Fzjph-Fzjmh)/dz
         + prod(op.f2V, v_collission_source)
         - prod(op.f2U, v_collission_source)
         + prod(op.f2U, u_collission_source)
         - prod(op.f2V, u_collission_source);
         //+ prod(op.f2V, me_nu_J)
         //+ prod(op.f2V, vsource)
         //- prod(op.f2U, me_nu_J)
         //- prod(op.f2U, vsource)
         //- prod(op.SG, xn)
         //+ boundaryconditions
         //+ prod(op.s2Np, isource)
         //- prod(op.s2Nn, isource)
         //+ prod(op.s2Tp, TpdivV)/3.0;
         //+ prod(op.s2Tn, TndivU)/3.0;
         //+ prod(op.s2Tp, nuin_dV2)/3.0
         //+ prod(op.s2Tn, nuni_dV2)/3.0
         //- two_thirds*prod(op.s2Tn, L)
         //+ prod(op.f2B, gNpxJxBoverN2) // Hall term source term
         //- prod(op.f2B,gradrescrossJ);
}

vcl_vec comfi::routines::computeRHS_RK4(const vcl_vec &xn, const double dt, const double t, const comfi::types::Operators &op, const comfi::types::BgData &bg)
{
  // RK-4
  const vcl_vec k1 = Re_MUSCL(xn,t,op,bg)*dt; //return xn+k1;
  const vcl_vec k2 = Re_MUSCL(xn+0.5*k1,t+0.5*dt,op,bg)*dt;
  const vcl_vec k3 = Re_MUSCL(xn+0.5*k2,t+0.5*dt,op,bg)*dt;
  const vcl_vec k4 = Re_MUSCL(xn+k3,t+dt,op,bg)*dt;

  return xn + (k1+2.0*k2+2.0*k3+k4)/6.0;
}
/*const vcl_vec mhdsim::routines::computeRHS_RK4SI(const vcl_vec &xn, const sp_mat_vcl &Ri, const double dt, const double t, const mhdsim::types::Operators &op, const BgData &bg)
{
  // RK-4
  const double damp = 0.5;
  const vcl_vec k0 = k(xn,t,dt,op,bg)*dt;
  const vcl_vec k1 = k(xn,t,dt,op,bg)*dt-damp*prod(Ri,xn); //return xn+k1;
  const vcl_vec k2 = k(xn+0.5*k1,t+0.5*dt,dt,op,bg)*dt;
  const vcl_vec k3 = k(xn+0.5*k2,t+0.5*dt,dt,op,bg)*dt;
  const vcl_vec k4 = k(xn+k3,t+dt,dt,op,bg)*dt;

  return xn + (k1+2.0*k2+2.0*k3+k4)/6.0;
}*/

vcl_vec comfi::routines::computeRHS_Euler(const vcl_vec &xn, const double dt, const double t, const comfi::types::Operators &op, const comfi::types::BgData &bg)
{
  // Simple Eulerian Steps
  const vcl_vec Re = comfi::routines::Re_MUSCL(xn, t, op, bg);
  const vcl_vec result = xn + Re*dt;

  //GLM
  const double a = 0.5;
  vcl_vec glm = prod(op.GLMs,result);
  glm *= std::exp(-a*op.ch/(ds/dt));

  return prod(op.ImGLM,result) + prod(op.s2GLM,glm);
}

vcl_vec comfi::routines::computeRHS_BDF2(const vcl_vec &xn,
                              const vcl_vec &xn1,
                              const vcl_sp_mat &Ri,
                              const double alpha,
                              const double beta,
                              const double dt,
                              const double t,
                              comfi::types::Operators &op,
                              const comfi::types::BgData &bg)
{
  // BDF2
  static double dtn1 = dt;

  const vcl_vec Rin = prod(Ri,xn);
  const vcl_vec Re = comfi::routines::Re_MUSCL(xn,t,op,bg);
  const vcl_vec xnpRedt = xn+Re*dt;
  const vcl_vec result = xnpRedt
                       + alpha*dt*(((xn-xn1)/dtn1)-Re)
                       + beta*dt*Rin;

  //GLM
  const double a = 0.1;
  vcl_vec glm = prod(op.GLMs,xnpRedt);
  glm *= std::exp(-a*op.ch/(ds/dt));

  dtn1=dt;
  return prod(op.ImGLM,result) + prod(op.s2GLM,glm);
}

void comfi::routines::topbc_driver(vcl_vec &Lxn, vcl_vec &Rxn, const double t, const comfi::types::Operators &op)
{
  uint ij = ind(0, nz-1);
  double v = 1.e-2*std::sin(2*arma::datum::pi*t/(t_0*2));
  Rxn(Vx+ij) = v;
  Rxn(Vp+ij) = v;
  Lxn(Vx+ij) = v;
  Lxn(Vp+ij) = v;
}

void comfi::routines::topbc_soler(vcl_vec &Lxn, vcl_vec &Rxn, const comfi::types::Operators &op)
{
  uint ij = ind(0, nz-1);
  Rxn(Vp+ij) = 0.0;
  Rxn(Vx+ij) = 0.0;
  Rxn(Up+ij) = 0.0;
  Rxn(Ux+ij) = 0.0;
  Lxn(Vp+ij) = 0.0;
  Lxn(Vx+ij) = 0.0;
  Lxn(Up+ij) = 0.0;
  Lxn(Ux+ij) = 0.0;
}

void comfi::routines::topbc_shock_tube(vcl_vec &Lxn, vcl_vec &Rxn, const comfi::types::Operators &op)
{
  vec top = zeros<vec>(num_of_elem);
  uint ij = ind(0, nz-1);
  top(n_p+ij) = 1.0;
  top(n_n+ij) = 1.0;
  top(T_n+ij) = 1.0/(gammamono-1.0);
  top(T_p+ij) = 1.0/(gammamono-1.0);
  vcl_vec top_vcl(num_of_elem);
  viennacl::fast_copy(top, top_vcl);
  Lxn = prod(op.ImTop, Lxn)+top_vcl;
  Rxn = prod(op.ImTop, Rxn)+top_vcl;
}

void comfi::routines::bottombc_soler(vcl_vec &Lxn, vcl_vec &Rxn, const comfi::types::Operators &op)
{
  uint ij = ind(0, 0);
  Lxn(Vp+ij) = 0.0;
  Lxn(Vx+ij) = 0.0;
  Lxn(Up+ij) = 0.0;
  Lxn(Ux+ij) = 0.0;
  Rxn(Vp+ij) = 0.0;
  Rxn(Vx+ij) = 0.0;
  Rxn(Up+ij) = 0.0;
  Rxn(Ux+ij) = 0.0;
}

void comfi::routines::bottombc_shock_tube(vcl_vec &Lxn, vcl_vec &Rxn, const comfi::types::Operators &op)
{
  vec bottom = zeros<vec>(num_of_elem);
  uint ij = ind(0, 0);
  bottom(n_p+ij) = 0.125;
  bottom(n_n+ij) = 0.125;
  bottom(T_n+ij) = 0.1/(gammamono-1.0);
  bottom(T_p+ij) = 0.1/(gammamono-1.0);
  vcl_vec bottom_vcl(num_of_elem);
  viennacl::fast_copy(bottom, bottom_vcl);
  Lxn = prod(op.ImBottom, Lxn)+bottom_vcl;
  Rxn = prod(op.ImBottom, Rxn)+bottom_vcl;
}

void comfi::routines::bottomBC(vcl_vec &Lxn, vcl_vec &Rxn, const double t, const comfi::types::Operators &op, const comfi::types::BgData &bg)
{
  double V = 0;
  const double V_b = 2000.0/V_0;
  const double w_p = 2*pi*1.65*1.e-3*t_0; //1.65mHz
  const double w_0 = 2*pi*1.0*1.e-3*t_0;
  const double w_m = 2*pi*10*t_0;
  const uint s = 100, m = 500;
  static std::mt19937::result_type seed = time(0);
  static auto mt_rand = std::bind(std::uniform_int_distribution<int>(0,s+m), std::mt19937(seed));
  vec bottom = zeros<vec>(num_of_elem);

  #pragma omp parallel for schedule(static)
  for(uint i=0; i<nx; i++)
  {
    for(uint k=0; k<s+m; k++)
    {
      if(k<s)
      {
        const double w=w_p*exp((10-k)*1.0/s*log(w_0/w_p));
        V += std::pow(w/w_p,five_sixths)*std::cos(w*t+2*pi*mt_rand()/(s+m));
      } else {
        const double w=w_m*exp((m-k+s)/(-m)*log(w_p/w_m));
        V += std::pow(w_p/w,five_sixths)*std::cos(w*t+2*pi*mt_rand()/(s+m));
      }
    }
    const int ij = ind(i, 0);
    bottom(n_p+ij) = Np0;
    bottom(n_n+ij) = Nn0;
    bottom(Vx+ij)  = 0.5*V_b*V*Np0;
    bottom(Ux+ij)  = 0.5*V_b*V*Nn0;
    bottom(Vp+ij)  = 0.5*V_b*V*Np0;
    bottom(Up+ij)  = 0.5*V_b*V*Nn0;
    bottom(T_p+ij) = T0;
    bottom(T_n+ij) = T0;
    //bottom(Bz+ij) = bg.BBz(i);
    V=0;
  }

  vcl_vec bottom_vcl(num_of_elem);
  viennacl::fast_copy(bottom,bottom_vcl);
  Lxn = prod(op.ImBottom,Lxn)+bottom_vcl;
  Rxn = prod(op.ImBottom,Rxn)+bottom_vcl;
}

void comfi::routines::bottomBCsquare(vcl_vec &Lxn, vcl_vec &Rxn, const double t, const comfi::types::Operators &op, const comfi::types::BgData &bg)
{
  double V_b = 25000.0/V_0;
  if (t > 0.1/t_0) { V_b = 0; }
  vec bottom = zeros<vec>(num_of_elem);

  #pragma omp parallel for schedule(static)
  for(uint i=0; i<nx; i++)
  {
    const int ij = ind(i,0);
    //bottom(n_p+ij) = Np0;
    //bottom(n_n+ij) = Nn0;
    //bottom(Vp+ij)  = V_b*Np0;
    //bottom(Up+ij)  = V_b*Nn0;
    bottom(T_p+ij) = T0;
    bottom(T_n+ij) = T0;
    bottom(Bz+ij) = bg.BBz(i);
  }

  vcl_vec bottom_vcl(num_of_elem);
  viennacl::fast_copy(bottom,bottom_vcl);
  Lxn = prod(op.ImBottom,Lxn)+bottom_vcl;
  Rxn = prod(op.ImBottom,Rxn)+bottom_vcl;
}

vcl_vec comfi::routines::Fx(const vcl_vec &xn, const vcl_vec &Npij, const comfi::types::Operators &op)
{
  vcl_vec F(num_of_elem);

  static const vcl_sp_mat Bf2Bxf = prod(op.s2f,op.fdotx);
  const vcl_vec Np = element_fabs(prod(op.Nps,xn));
  const vcl_vec Nn = element_fabs(prod(op.Nns,xn));
  const vcl_vec Tp = element_fabs(prod(op.Tps,xn));
  const vcl_vec Tn = element_fabs(prod(op.Tns,xn));
  const vcl_vec Bf = prod(op.Bf,xn);
  const vcl_vec Bx = prod(op.fdotx,Bf);
  const vcl_vec hB2 = 0.5*comfi::routines::dot_prod(Bf,Bf,op); //mag pressure scalar col vec
  const vcl_vec Bxf = prod(Bf2Bxf,Bf);
  const vcl_vec BBx = element_prod(Bxf,Bf);
  const vcl_vec Vf = element_div(prod(op.Vf,xn),prod(op.s2f,Np));
  const vcl_vec VBx = element_prod(Bxf,Vf);
  const vcl_vec Jf = prod(op.curl,Bf);
  const vcl_vec Jxf = prod(Bf2Bxf, Jf);
  const vcl_vec BcrossJ = comfi::routines::cross_prod(Bf,Jf,op);
  const vcl_vec Pp = element_prod(Tp,Np);
  const vcl_vec Pn = element_prod(Tn,Nn);
  const vcl_vec res = comfi::sol::resistivity(Np,Nn,Tp,Tn);
  const vcl_vec resBcrossJ = element_div(element_prod(res,prod(op.fdotx,BcrossJ)),Npij);
  const vcl_vec glm = prod(op.GLMs,xn);
  const vcl_vec BxJ_JxB = element_prod(Bxf, Jf) - element_prod(Jxf, Bf);
  const vcl_vec BxJ_JxBoverN = element_div(BxJ_JxB, q*prod(op.s2f, Npij));

  F = element_prod(element_div(prod(op.PEVx,xn),prod(op.PN,xn)),xn) // local speed eigen values
    + prod(op.s2Vx, Pp)              // thermal pressure
    + prod(op.s2Ux, Pn)              // thermal pressure
    + prod(op.s2Vx, hB2)            // magnetic pressure
    - prod(op.pFBB, BBx)             // magnetic tension
    - prod(op.FxVB, VBx)            // second eig induction
    + prod(op.s2Bx, glm)             // lagrange multiplier constraint
    + prod(op.f2B, BxJ_JxBoverN)
    - two_thirds*prod(op.s2Tp,resBcrossJ)    //ohmic heating
    + prod(op.s2GLM,Bx)*op.ch*op.ch;

  return F;
}

vcl_vec comfi::routines::Fz(const vcl_vec &xn, const vcl_vec &Npij, const comfi::types::Operators &op)
{
  vcl_vec F(num_of_elem);

  static const vcl_sp_mat Bf2Bzf = prod(op.s2f, op.fdotz);
  const vcl_vec Np = element_fabs(prod(op.Nps, xn));
  const vcl_vec Nn = element_fabs(prod(op.Nns, xn));
  const vcl_vec Tp = element_fabs(prod(op.Tps, xn));
  const vcl_vec Tn = element_fabs(prod(op.Tns, xn));
  const vcl_vec Bf = prod(op.Bf, xn);
  const vcl_vec Bz = prod(op.fdotz, Bf);
  const vcl_vec Bzf = prod(Bf2Bzf, Bf);
  const vcl_vec hB2 = 0.5*comfi::routines::dot_prod(Bf,Bf,op); //mag pressure scalar col vec
  const vcl_vec BBz = element_prod(Bzf,Bf);
  const vcl_vec Vf = element_div(prod(op.Vf, xn), prod(op.s2f, Np));
  const vcl_vec VBz = element_prod(Bzf, Vf);
  const vcl_vec Jf = prod(op.curl,Bf);
  const vcl_vec Jzf = prod(Bf2Bzf, Jf);
  const vcl_vec BcrossJ = comfi::routines::cross_prod(Bf,Jf,op);
  const vcl_vec Pp = element_prod(Tp, Np);
  const vcl_vec Pn = element_prod(Tn, Nn);

  /*
  const vcl_vec Uf = element_div(prod(op.Uf, xn), prod(op.s2f, Nn));
  vcl_vec k_e0 = 0.5*element_prod(prod(op.s2f, Nn), Uf);
  vcl_vec k_e = mhdsim::routines::dot_prod(k_e0, Uf, op);
  const vcl_vec Pn  = element_fabs((gammamono-1.0)*(Tn-k_e));
  const vcl_vec PnUz = element_prod(Pn, prod(op.fdotz, Uf));
  */
  //const vcl_vec res = mhdsim::sol::resistivity(Np,Nn,Tp,Tn);
  //const vcl_vec resBcrossJ = element_div(element_prod(res, prod(op.fdotz,BcrossJ)), Npij);
  const vcl_vec glm = prod(op.GLMs,xn);
  //const vcl_vec BzJ_JzB = element_prod(Bzf, Jf) - element_prod(Jzf, Bf);
  //const vcl_vec BzJ_JzBoverN = element_div(BzJ_JzB, q*prod(op.s2f, Npij));

  F = element_prod(element_div(prod(op.PEVz,xn), prod(op.PN, xn)), xn) // local speed eigen values

    //+ prod(op.s2Tn, PnUz)

    + prod(op.s2Vz, Pp)              // thermal pressure ion
    + prod(op.s2Uz, Pn)              // thermal pressure neutral
    + prod(op.s2Vz, hB2)             // magnetic pressure
    - prod(op.pFBB, BBz)             // magnetic tension
    - prod(op.FzVB, VBz);             // second eign induction
    //+ prod(op.s2Bz, glm);             // lagrange multiplier constraint
    //+ prod(op.f2B, BzJ_JzBoverN) // Hall term
    //- two_thirds*prod(op.s2Tp, resBcrossJ)    // joule heating
    //+ prod(op.s2GLM, Bz)*op.ch*op.ch;

    return F;
}

sp_mat buildPoisson()
{
  const uint nnzp = 5;
  umat  Avi = zeros<umat>(nnzp, num_of_grid);
  umat  Avj = zeros<umat>(nnzp, num_of_grid);
  mat   Avv = zeros<mat> (nnzp, num_of_grid);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i
    const unsigned int  j=(index)/nx; //find j
    const int           sij=inds(i,j);

    int                 sijm1=inds(i,j-1);
    int                 sijp1=inds(i,j+1);
    int                 sip1j=inds(i+1,j);
    int                 sim1j=inds(i-1,j);
    //BC
    if (j==0) { sijm1=sij; } else
    if (j==nz-1) { sijp1=sij; }
    if (i==0) { sim1j=sij; } else
    if (i==nx-1) { sip1j=sij; }

    // Build sparse bulk vectors
    int p=-1;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p=p+1; vi(p)=sij; vj(p)=sij;
    vv(p) = -2.0/dz2-2.0/dx2;
    p=p+1; vi(p)=sij; vj(p)=sijp1;
    vv(p) = 1.0/dz2;
    p=p+1; vi(p)=sij; vj(p)=sijm1;
    vv(p) = 1.0/dz2;
    p=p+1; vi(p)=sij; vj(p)=sip1j;
    vv(p) = 1.0/dx2;
    p=p+1; vi(p)=sij; vj(p)=sim1j;
    vv(p) = 1.0/dx2;
    // Collect sparse values and locations
    Avi(span::all,index)=vi;
    Avj(span::all,index)=vj;
    Avv(span::all,index)=vv;
  }
  // Reorder to create sparse matrix

  return comfi::util::syncSpMat(Avi,Avj,Avv,num_of_grid,num_of_grid); //reorder due to parallel construction
}

sp_mat buildJacobi()
{
  const uint nnzp= 4;
  umat  Avi = zeros<umat>(nnzp,num_of_grid);
  umat  Avj = zeros<umat>(nnzp,num_of_grid);
  mat   Avv = zeros<mat> (nnzp,num_of_grid);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i
    const unsigned int  j=(index)/nx; //find j
    const int           sij=inds(i,j);

    int                 sijm1=inds(i,j-1);
    int                 sijp1=inds(i,j+1);
    int                 sip1j=inds(i+1,j);
    int                 sim1j=inds(i-1,j);
    //BC
    if (j==0) { sijm1=sij; } else
    if (j==nz-1) { sijp1=sij; }
    if (i==0) { sim1j=sij; } else
    if (i==nx-1) { sip1j=sij; }

    // Build sparse bulk vectors
    int p=-1;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p=p+1; vi(p)=sij; vj(p)=sijp1;
    vv(p) = dx2;
    p=p+1; vi(p)=sij; vj(p)=sijm1;
    vv(p) = dx2;
    p=p+1; vi(p)=sij; vj(p)=sip1j;
    vv(p) = dz2;
    p=p+1; vi(p)=sij; vj(p)=sim1j;
    vv(p) = dz2;
    // Collect sparse values and locations
    Avi(span::all,index)=vi;
    Avj(span::all,index)=vj;
    Avv(span::all,index)=vv;
  }
  // Reorder to create sparse matrix

  return comfi::util::syncSpMat(Avi,Avj,Avv,num_of_grid,num_of_grid); //reorder due to parallel construction
}

vcl_vec comfi::routines::cz_max(const vcl_vec &xn, const comfi::types::Operators &op) {
  const vcl_vec v = element_div(prod(op.PEVz, xn), prod(op.PN, xn));
  return fast_speed_z(xn, op) + element_fabs(v);
}

vcl_vec comfi::routines::cx_max(const vcl_vec &xn, const comfi::types::Operators &op) {
  const vcl_vec v = element_div(prod(op.PEVx, xn), prod(op.PN, xn));
  return fast_speed_x(xn, op) + element_fabs(v);
}

vcl_vec comfi::routines::fast_speed_x(const vcl_vec &xn, const comfi::types::Operators &op)
{
  const vcl_vec Bf = prod(op.Bf, xn);
  const vcl_vec Np = element_fabs(prod(op.Nps, xn));
  const vcl_vec Nn = element_fabs(prod(op.Nns, xn));
  const vcl_vec Tp = element_fabs(prod(op.Tps, xn));
  const vcl_vec Tn = element_fabs(prod(op.Tns, xn));
  const vcl_vec Vf = element_div(prod(op.Vf, xn), prod(op.s2f, Np));
  const vcl_vec Vx = prod(op.fdotx, Vf);
  const vcl_vec Uf = element_div(prod(op.Uf, xn), prod(op.s2f, Nn));
  const vcl_vec Ux = prod(op.fdotx, Uf);
  const vcl_vec B2 = comfi::routines::dot_prod(Bf, Bf, op);
  const vcl_vec Pp  = element_prod(Tp, Np);
  const vcl_vec Pn  = element_prod(Tn,Nn);

  /*
  vcl_vec k_e0 = 0.5*element_prod(prod(op.s2f, Nn), Uf);
  vcl_vec k_e = mhdsim::routines::dot_prod(k_e0, Uf, op);
  const vcl_vec Pn  = element_fabs((gammamono-1.0)*(Tn - k_e));
  */

  const vcl_vec cps2 = gammamono*(element_div(Pp, Np));
  const vcl_vec cps = element_sqrt(cps2);
  const vcl_vec cns2 = gammamono*(element_div(Pn, Nn));
  const vcl_vec cns = element_sqrt(cns2);
  const vcl_vec ca2 = element_div(B2, Np);
  const vcl_vec cax = element_div(prod(op.fdotx, Bf), element_sqrt(Np));
  const vcl_vec cpsca = element_prod(cps, cax);
  const vcl_vec cpsca2= element_prod(cpsca, cpsca);

  const vcl_vec cp = 0.5*(element_sqrt(2.0*(cps2 + ca2 + element_sqrt(element_prod(cps2+ca2,cps2+ca2)-4.0*cpsca2))));
  const vcl_vec cn = cns;

  const vcl_vec cpf = prod(op.s2f, cp);
  const vcl_vec cnf = prod(op.s2f, cn);

  return prod(op.s2Np, cp)
        +prod(op.s2Nn, cn)
        +prod(op.f2V, cpf)
        +prod(op.f2U, cnf)
        +prod(op.s2Tp, cp)
        +prod(op.s2Tn, cn)
        +prod(op.f2B, cpf)
        +prod(op.s2GLM, cp);
}

vcl_vec comfi::routines::fast_speed_z(const vcl_vec &xn, const comfi::types::Operators &op)
{
  const vcl_vec Bf = prod(op.Bf,xn);
  const vcl_vec Np = element_fabs(prod(op.Nps,xn));
  const vcl_vec Nn = element_fabs(prod(op.Nns,xn));
  const vcl_vec Tp = element_fabs(prod(op.Tps,xn));
  const vcl_vec Tn = element_fabs(prod(op.Tns,xn));
  const vcl_vec Vf = element_div(prod(op.Vf,xn),prod(op.s2f,Np));
  const vcl_vec Vz = prod(op.fdotz, Vf);
  const vcl_vec Uf = element_div(prod(op.Uf,xn),prod(op.s2f,Nn));
  const vcl_vec Uz = prod(op.fdotz, Uf);
  const vcl_vec B2 = comfi::routines::dot_prod(Bf,Bf,op);
  const vcl_vec Pp  = element_prod(Tp, Np);
  const vcl_vec Pn  = element_prod(Tn, Nn);

  /*
  vcl_vec k_e0 = 0.5*element_prod(prod(op.s2f, Nn), Uf);
  vcl_vec k_e = mhdsim::routines::dot_prod(k_e0, Uf, op);
  const vcl_vec Pn  = element_fabs((gammamono-1.0)*(Tn - k_e));
  */

  const vcl_vec cps2 = gammamono*(element_div(Pp,Np));
  const vcl_vec cps = element_sqrt(cps2);
  const vcl_vec cns2 = gammamono*(element_div(Pn,Nn));
  const vcl_vec cn = element_sqrt(cns2);
  const vcl_vec ca2 = element_div(B2,Np);
  const vcl_vec caz = element_div(prod(op.fdotz,Bf), element_sqrt(Np));
  const vcl_vec cpsca = element_prod(cps,caz);
  const vcl_vec cpsca2= element_prod(cpsca,cpsca);

  const vcl_vec cp = 0.5*(element_sqrt(2.0*(cps2 + ca2 + element_sqrt(element_prod(cps2+ca2, cps2+ca2)-4.0*cpsca2))));

  const vcl_vec cpf = prod(op.s2f, cp);
  const vcl_vec cnf = prod(op.s2f, cn);
  const vcl_vec cnsf = prod(op.s2f, cn);

  return prod(op.s2Np, cp)
        +prod(op.s2Nn, cn)
        +prod(op.f2V, cpf)
        +prod(op.f2U, cnf)
        +prod(op.s2Tp, cp)
        +prod(op.s2Tn, cn)
        +prod(op.f2B, cpf)
        +prod(op.s2GLM, cp);
}


vcl_vec comfi::routines::cross_prod(const vcl_vec &f1, const vcl_vec &f2, const comfi::types::Operators &op)
{
  return element_prod(prod(op.cross1, f1), prod(op.cross2, f2)) - element_prod(prod(op.cross1, f2), prod(op.cross2, f1));
}

vcl_vec comfi::routines::dot_prod(const vcl_vec &f1, const vcl_vec &f2, const comfi::types::Operators &op)
{
  const vcl_vec f = element_prod(f1,f2);
  return prod(op.f2s,f);
}

vcl_vec comfi::routines::polyval(const arma::vec &p, const vcl_vec &x)
{
  vcl_vec b = viennacl::zero_vector<double>(x.size());

  for (int i = 0; i < p.size(); i++)
  {
    const vcl_vec a = viennacl::scalar_vector<double>(x.size(), p(i));
    b = a + element_prod(b, x);
  }

  return b;
}

vcl_mat comfi::routines::fluxl(const vcl_mat &r) {
  static const vcl_mat ones = viennacl::scalar_matrix<double>(r.size1(), r.size2(), 1.0);
  const vcl_mat r2 = element_prod(r, r);
  return 1.5*element_div(r2+r,r2+r+ones);
}
