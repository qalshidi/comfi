#include <armadillo>
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/forwards.h"
#include "comfi.hpp"
#include "params.h"

using namespace viennacl::linalg;
using namespace arma;

sp_mat comfi::routines::computeRi(const vcl_vec &xn_vcl, const comfi::types::Operators &op)
{
  vec xn(num_of_elem);
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
    const double Tpij   = std::abs(xn(E_p+ij));
    const double Nnij   = std::abs(xn(n_n+ij));
    const double Tnij   = std::abs(xn(E_n+ij));
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

vcl_mat comfi::routines::Re_MUSCL(const vcl_mat &xn, comfi::types::Context &ctx)
{
  const vcl_mat xn_ip1 = comfi::operators::ip1(xn, ctx);
  const vcl_mat xn_im1 = comfi::operators::im1(xn, ctx);
  const vcl_mat xn_jp1 = comfi::operators::jp1(xn, ctx);
  const vcl_mat xn_jm1 = comfi::operators::jm1(xn, ctx);

  const vcl_mat dxn_iph = xn_ip1-xn;
  const vcl_mat dxn_imh = xn-xn_im1;
  const vcl_mat dxn_jph = xn_jp1-xn;
  const vcl_mat dxn_jmh = xn-xn_jm1;

  /*
  viennacl::ocl::program & fluxl_prog  = viennacl::ocl::current_context().get_program("fluxl");
  viennacl::ocl::kernel  & fluxl = fluxl_prog.get_kernel("fluxl");

  const vcl_mat r_i   = element_div(dxn_imh, dxn_iph);
  const vcl_mat r_ip1 = element_div(dxn_iph, (comfi::operators::ip1(xn_ip1, ctx)-xn_ip1));
  const vcl_mat r_im1 = element_div((xn_im1-comfi::operators::im1(xn_im1, ctx)), dxn_imh);
  const vcl_mat r_j   = element_div(dxn_jmh, dxn_jph);
  const vcl_mat r_jp1 = element_div(dxn_jph, (comfi::operators::jp1(xn_jp1, ctx)-xn_jp1));
  const vcl_mat r_jm1 = element_div((xn_jm1-comfi::operators::jm1(xn_jm1, ctx)), dxn_jmh);
  vcl_mat phi_i(xn.size1(), xn.size2()), phi_ip1(xn.size1(), xn.size2()), phi_im1(xn.size1(), xn.size2()),
          phi_j(xn.size1(), xn.size2()), phi_jp1(xn.size1(), xn.size2()), phi_jm1(xn.size1(), xn.size2());
  viennacl::ocl::enqueue(fluxl(r_i, phi_i,
                               cl_uint(r_i.size1()*r_i.size2())));
  viennacl::ocl::enqueue(fluxl(r_ip1, phi_ip1,
                               cl_uint(r_i.size1()*r_i.size2())));
  viennacl::ocl::enqueue(fluxl(r_im1, phi_im1,
                               cl_uint(r_i.size1()*r_i.size2())));
  viennacl::ocl::enqueue(fluxl(r_j, phi_j,
                               cl_uint(r_i.size1()*r_i.size2())));
  viennacl::ocl::enqueue(fluxl(r_jp1, phi_jp1,
                               cl_uint(r_i.size1()*r_i.size2())));
  viennacl::ocl::enqueue(fluxl(r_jm1, phi_jm1,
                               cl_uint(r_i.size1()*r_i.size2())));

  vcl_mat Lxn_iph = xn     + 0.5*element_prod(phi_i, dxn_imh);
  vcl_mat Lxn_imh = xn_im1 + 0.5*element_prod(phi_im1, dxn_imh);
  vcl_mat Rxn_iph = xn_ip1 - 0.5*element_prod(phi_ip1, dxn_iph);
  vcl_mat Rxn_imh = xn     - 0.5*element_prod(phi_i, dxn_iph);
  vcl_mat Lxn_jph = xn     + 0.5*element_prod(phi_j, dxn_jmh);
  vcl_mat Lxn_jmh = xn_jm1 + 0.5*element_prod(phi_jm1, dxn_jmh);
  vcl_mat Rxn_jph = xn_jp1 - 0.5*element_prod(phi_jp1, dxn_jph);
  vcl_mat Rxn_jmh = xn     - 0.5*element_prod(phi_j, dxn_jph);
  */

  const vcl_mat eps = viennacl::scalar_matrix<double>(xn.size1(), xn.size2(), 1.e-100);
  const vcl_mat r_i   = element_div(dxn_imh, dxn_iph+eps);
  const vcl_mat r_ip1 = element_div(dxn_iph, (comfi::operators::ip1(xn_ip1, ctx)-xn_ip1)+eps);
  const vcl_mat r_im1 = element_div((xn_im1-comfi::operators::im1(xn_im1, ctx)), dxn_imh+eps);
  const vcl_mat r_j   = element_div(dxn_jmh, dxn_jph+eps);
  const vcl_mat r_jp1 = element_div(dxn_jph, (comfi::operators::jp1(xn_jp1, ctx)-xn_jp1)+eps);
  const vcl_mat r_jm1 = element_div((xn_jm1-comfi::operators::jm1(xn_jm1, ctx)), dxn_jmh+eps);
  //extrapolated cell edge variables
  vcl_mat Lxn_iph = xn     + 0.5*element_prod(comfi::routines::fluxl(r_i), dxn_imh);
  vcl_mat Lxn_imh = xn_im1 + 0.5*element_prod(comfi::routines::fluxl(r_im1), dxn_imh);
  vcl_mat Rxn_iph = xn_ip1 - 0.5*element_prod(comfi::routines::fluxl(r_ip1), dxn_iph);
  vcl_mat Rxn_imh = xn     - 0.5*element_prod(comfi::routines::fluxl(r_i), dxn_iph);
  vcl_mat Lxn_jph = xn     + 0.5*element_prod(comfi::routines::fluxl(r_j), dxn_jmh);
  vcl_mat Lxn_jmh = xn_jm1 + 0.5*element_prod(comfi::routines::fluxl(r_jm1), dxn_jmh);
  vcl_mat Rxn_jph = xn_jp1 - 0.5*element_prod(comfi::routines::fluxl(r_jp1), dxn_jph);
  vcl_mat Rxn_jmh = xn     - 0.5*element_prod(comfi::routines::fluxl(r_j), dxn_jph);
   // BOUNDARY CONDITIONS
  //mhdsim::routines::bottomBC(Lxn_jmh,Rxn_jmh,t,op,bg);
  //comfi::routines::bottombc_shock_tube(Lxn_jmh, Rxn_jmh, ctx);
  //comfi::routines::topbc_shock_tube(Lxn_jph, Rxn_jph, ctx);
  //comfi::routines::topbc_soler(Lxn_jph, Rxn_jph, op);
  //mhdsim::routines::topbc_driver(Lxn_jph, Rxn_jph, t, op);
  //comfi::routines::bottombc_soler(Lxn_jmh, Rxn_jmh, op);

  // Fast mode speed eigenvalues
  vcl_mat Leig_iph_p = comfi::routines::fast_speed_x_mat(Lxn_iph, ctx);
  vcl_mat Reig_iph_p = comfi::routines::fast_speed_x_mat(Rxn_iph, ctx);
  vcl_mat Leig_jph_p = comfi::routines::fast_speed_z_mat(Lxn_jph, ctx);
  vcl_mat Reig_jph_p = comfi::routines::fast_speed_z_mat(Rxn_jph, ctx);
  vcl_mat Leig_imh_p = comfi::routines::fast_speed_x_mat(Lxn_imh, ctx);
  vcl_mat Reig_imh_p = comfi::routines::fast_speed_x_mat(Rxn_imh, ctx);
  vcl_mat Leig_jmh_p = comfi::routines::fast_speed_z_mat(Lxn_jmh, ctx);
  vcl_mat Reig_jmh_p = comfi::routines::fast_speed_z_mat(Rxn_jmh, ctx);
  viennacl::ocl::program & eig_prog  = viennacl::ocl::current_context().get_program("element_max");
  viennacl::ocl::kernel  & element_max = eig_prog.get_kernel("element_max");

  vcl_mat a_imh_p(ctx.num_of_grid(), 1);
  viennacl::ocl::enqueue(element_max(Leig_imh_p, Reig_imh_p,
                                     a_imh_p,
                                     cl_uint(Leig_imh_p.size1())));
  vcl_mat a_iph_p(ctx.num_of_grid(), 1);
  viennacl::ocl::enqueue(element_max(Leig_iph_p, Reig_iph_p,
                                     a_iph_p,
                                     cl_uint(Leig_iph_p.size1())));
  vcl_mat a_jmh_p(ctx.num_of_grid(), 1);
  viennacl::ocl::enqueue(element_max(Leig_jmh_p, Reig_jmh_p,
                                     a_jmh_p,
                                     cl_uint(Leig_jmh_p.size1())));
  vcl_mat a_jph_p(ctx.num_of_grid(), 1);
  viennacl::ocl::enqueue(element_max(Leig_jph_p, Reig_jph_p,
                                     a_jph_p,
                                     cl_uint(Leig_jph_p.size1())));
  /*
  Leig_iph_p = element_fabs(element_div(ctx.v_NVx(Lxn_iph), ctx.v_Np(Lxn_iph)));
  viennacl::ocl::enqueue(element_max(Leig_iph_p, a_iph_p,
                                     a_iph_p,
                                     cl_uint(Leig_iph_p.size1())));

  Reig_iph_p = element_fabs(element_div(ctx.v_NVx(Rxn_iph), ctx.v_Np(Rxn_iph)));
  viennacl::ocl::enqueue(element_max(Reig_iph_p, a_iph_p,
                                     a_iph_p,
                                     cl_uint(Reig_iph_p.size1())));

  Leig_jph_p = element_fabs(element_div(ctx.v_NVz(Lxn_jph), ctx.v_Np(Lxn_jph)));
  viennacl::ocl::enqueue(element_max(Leig_jph_p, a_jph_p,
                                     a_jph_p,
                                     cl_uint(Leig_jph_p.size1())));

  Reig_jph_p = element_fabs(element_div(ctx.v_NVz(Rxn_jph), ctx.v_Np(Rxn_jph)));
  viennacl::ocl::enqueue(element_max(Reig_jph_p, a_jph_p,
                                     a_jph_p,
                                     cl_uint(Reig_jph_p.size1())));

  Leig_imh_p = element_fabs(element_div(ctx.v_NVx(Lxn_imh), ctx.v_Np(Lxn_imh)));
  viennacl::ocl::enqueue(element_max(Leig_imh_p, a_imh_p,
                                     a_imh_p,
                                     cl_uint(Leig_imh_p.size1())));

  Reig_imh_p = element_fabs(element_div(ctx.v_NVx(Rxn_imh), ctx.v_Np(Rxn_imh)));
  viennacl::ocl::enqueue(element_max(Reig_imh_p, a_imh_p,
                                     a_imh_p,
                                     cl_uint(Reig_imh_p.size1())));

  Leig_jmh_p = element_fabs(element_div(ctx.v_NVz(Lxn_jmh), ctx.v_Np(Lxn_jmh)));
  viennacl::ocl::enqueue(element_max(Leig_jmh_p, a_jmh_p,
                                     a_jmh_p,
                                     cl_uint(Leig_jmh_p.size1())));

  Reig_jmh_p = element_fabs(element_div(ctx.v_NVz(Rxn_jmh), ctx.v_Np(Rxn_jmh)));
  viennacl::ocl::enqueue(element_max(Reig_jmh_p, a_jmh_p,
                                     a_jmh_p,
                                     cl_uint(Reig_jmh_p.size1())));
                                     */

  vcl_mat Leig_iph_n = comfi::routines::sound_speed_neutral_mat(Lxn_iph, ctx);
  vcl_mat Reig_iph_n = comfi::routines::sound_speed_neutral_mat(Rxn_iph, ctx);
  vcl_mat Leig_jph_n = comfi::routines::sound_speed_neutral_mat(Lxn_jph, ctx);
  vcl_mat Reig_jph_n = comfi::routines::sound_speed_neutral_mat(Rxn_jph, ctx);
  vcl_mat Leig_imh_n = comfi::routines::sound_speed_neutral_mat(Lxn_imh, ctx);
  vcl_mat Reig_imh_n = comfi::routines::sound_speed_neutral_mat(Rxn_imh, ctx);
  vcl_mat Leig_jmh_n = comfi::routines::sound_speed_neutral_mat(Lxn_jmh, ctx);
  vcl_mat Reig_jmh_n = comfi::routines::sound_speed_neutral_mat(Rxn_jmh, ctx);
  vcl_mat a_imh_n(ctx.num_of_grid(), 1);
  if (ctx.bc_left() != comfi::types::DIMENSIONLESS) {
    viennacl::ocl::enqueue(element_max(Leig_imh_n, Reig_imh_n,
                                       a_imh_n,
                                       cl_uint(Leig_imh_n.size1())));
  }
  vcl_mat a_iph_n(ctx.num_of_grid(), 1);
  if (ctx.bc_right() != comfi::types::DIMENSIONLESS) {
    viennacl::ocl::enqueue(element_max(Leig_iph_n, Reig_iph_n,
                                       a_iph_n,
                                       cl_uint(Leig_iph_n.size1())));
  }
  vcl_mat a_jmh_n(ctx.num_of_grid(), 1);
  if (ctx.bc_down() != comfi::types::DIMENSIONLESS) {
    viennacl::ocl::enqueue(element_max(Leig_jmh_n, Reig_jmh_n,
                                       a_jmh_n,
                                       cl_uint(Leig_jmh_n.size1())));
  }
  vcl_mat a_jph_n(ctx.num_of_grid(), 1);
  if (ctx.bc_up() != comfi::types::DIMENSIONLESS) {
    viennacl::ocl::enqueue(element_max(Leig_jph_n, Reig_jph_n,
                                       a_jph_n,
                                       cl_uint(Leig_jph_n.size1())));
  }

  /*
  Reig_jph_n = element_fabs(element_div(ctx.v_NUz(Rxn_jph), ctx.v_Nn(Rxn_jph)));
  viennacl::ocl::enqueue(element_max(Reig_jph_n, a_jph_n,
                                     a_jph_n,
                                     cl_uint(Reig_jph_n.size1())));

  Reig_jmh_n = element_fabs(element_div(ctx.v_NUz(Rxn_jmh), ctx.v_Nn(Rxn_jmh)));
  viennacl::ocl::enqueue(element_max(Reig_jmh_n, a_jmh_n,
                                     a_jmh_n,
                                     cl_uint(Reig_jmh_n.size1())));

  Reig_imh_n = element_fabs(element_div(ctx.v_NUx(Rxn_imh), ctx.v_Nn(Rxn_imh)));
  viennacl::ocl::enqueue(element_max(Reig_imh_n, a_imh_n,
                                     a_imh_n,
                                     cl_uint(Reig_imh_n.size1())));

  Reig_iph_n = element_fabs(element_div(ctx.v_NUx(Rxn_iph), ctx.v_Nn(Rxn_iph)));
  viennacl::ocl::enqueue(element_max(Reig_iph_n, a_iph_n,
                                     a_iph_n,
                                     cl_uint(Reig_iph_n.size1())));

  Leig_jph_n = element_fabs(element_div(ctx.v_NUz(Lxn_jph), ctx.v_Nn(Lxn_jph)));
  viennacl::ocl::enqueue(element_max(Leig_jph_n, a_jph_n,
                                     a_jph_n,
                                     cl_uint(Leig_jph_n.size1())));

  Leig_jmh_n = element_fabs(element_div(ctx.v_NUz(Lxn_jmh), ctx.v_Nn(Lxn_jmh)));
  viennacl::ocl::enqueue(element_max(Leig_jmh_n, a_jmh_n,
                                     a_jmh_n,
                                     cl_uint(Leig_jmh_n.size1())));

  Leig_imh_n = element_fabs(element_div(ctx.v_NUx(Lxn_imh), ctx.v_Nn(Lxn_imh)));
  viennacl::ocl::enqueue(element_max(Leig_imh_n, a_imh_n,
                                     a_imh_n,
                                     cl_uint(Leig_imh_n.size1())));

  Leig_iph_n = element_fabs(element_div(ctx.v_NUx(Lxn_iph), ctx.v_Nn(Lxn_iph)));
  viennacl::ocl::enqueue(element_max(Leig_iph_n, a_iph_n,
                                     a_iph_n,
                                     cl_uint(Leig_iph_n.size1())));
*/
  static vcl_mat p = viennacl::zero_matrix<double>(1, ctx.num_of_eq());
  static vcl_mat n = viennacl::zero_matrix<double>(1, ctx.num_of_eq());
  static bool unit_vecs_unfilled = true;
  if (unit_vecs_unfilled) {
    p(0, n_p) = 1.0;
    p(0, Vx) = 1.0;
    p(0, Vz) = 1.0;
    p(0, Vp) = 1.0;
    p(0, Bx) = 1.0;
    p(0, Bz) = 1.0;
    p(0, Bp) = 1.0;
    p(0, E_p) = 1.0;
    p(0, GLM) = 1.0;
    n(0, n_n) = 1.0;
    n(0, Ux) = 1.0;
    n(0, Uz) = 1.0;
    n(0, Up) = 1.0;
    n(0, E_n) = 1.0;
    unit_vecs_unfilled = false;
  }

  // LAX-FRIEDRICHS FLUX
  /*
  const vcl_vec Fximh = 0.5*(mhdsim::routines::Fx(Lxn_imh, sNp, op)+mhdsim::routines::Fx(Rxn_imh, sNp, op))
                      - element_prod(a_imh,(Rxn_imh - Lxn_imh));
  const vcl_vec Fxiph = 0.5*(mhdsim::routines::Fx(Lxn_iph, sNp, op)+mhdsim::routines::Fx(Rxn_iph, sNp, op))
                      - element_prod(a_iph,(Rxn_iph-Lxn_iph));
  */
  vcl_mat a_jmh = prod(a_jmh_p, p)
                 +prod(a_jmh_n, n);
  const vcl_mat Fzjmh = 0.5*(comfi::routines::Fz(Lxn_jmh, xn, ctx)+comfi::routines::Fz(Rxn_jmh, xn, ctx))
                             -element_prod(a_jmh, (Rxn_jmh-Lxn_jmh));

  vcl_mat a_jph = prod(a_jph_p, p)
                 +prod(a_jph_n, n);
  const vcl_mat Fzjph = 0.5*(comfi::routines::Fz(Lxn_jph, xn, ctx)+comfi::routines::Fz(Rxn_jph, xn, ctx))
                             -element_prod(a_jph, (Rxn_jph-Lxn_jph));

  return //-1.0*(Fxiph-Fximh)/dx
         -1.0*(Fzjph-Fzjmh)/dz;
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

/*
vcl_vec comfi::routines::computeRHS_RK4(const vcl_vec &xn, const double dt, const double t, const comfi::types::Operators &op, const comfi::types::BgData &bg)
{
  // RK-4
  const vcl_vec k1 = Re_MUSCL(xn,t,op,bg)*dt; //return xn+k1;
  const vcl_vec k2 = Re_MUSCL(xn+0.5*k1,t+0.5*dt,op,bg)*dt;
  const vcl_vec k3 = Re_MUSCL(xn+0.5*k2,t+0.5*dt,op,bg)*dt;
  const vcl_vec k4 = Re_MUSCL(xn+k3,t+dt,op,bg)*dt;

  return xn + (k1+2.0*k2+2.0*k3+k4)/6.0;
}
*/
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

vcl_mat comfi::routines::computeRHS_Euler(const vcl_mat &xn, comfi::types::Context &ctx)
{
  // Simple Eulerian Steps
  vcl_mat result = xn + comfi::routines::Re_MUSCL(xn, ctx)*ctx.dt();

  // GLM exact solution
  const double a = 0.5;
  const double ch = ds/ctx.dt();
  ctx.v_GLM(result) *= std::exp(-a*ch/(ds/ctx.dt()));

  return result;
}

/*
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
*/

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
  top(E_n+ij) = 1.0/(gammamono-1.0);
  top(E_p+ij) = 1.0/(gammamono-1.0);
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
  bottom(E_n+ij) = 0.1/(gammamono-1.0);
  bottom(E_p+ij) = 0.1/(gammamono-1.0);
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
    bottom(E_p+ij) = T0;
    bottom(E_n+ij) = T0;
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
    bottom(E_p+ij) = T0;
    bottom(E_n+ij) = T0;
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

vcl_mat comfi::routines::Fz(const vcl_mat &xn, const vcl_mat &xn_ij, comfi::types::Context &ctx)
{
  vcl_mat F = viennacl::zero_matrix<double>(xn.size1(), xn.size2());

  vcl_mat V_x(xn.size1(), 1);
  vcl_mat U_x(xn.size1(), 1);
  vcl_mat V_z(xn.size1(), 1);
  vcl_mat U_z(xn.size1(), 1);
  vcl_mat V_p(xn.size1(), 1);
  vcl_mat U_p(xn.size1(), 1);
  V_x = element_div(ctx.v_NVx(xn), ctx.v_Np(xn));
  U_x = element_div(ctx.v_NUx(xn), ctx.v_Nn(xn));
  V_z = element_div(ctx.v_NVz(xn), ctx.v_Np(xn));
  U_z = element_div(ctx.v_NUz(xn), ctx.v_Nn(xn));
  V_p = element_div(ctx.v_NVp(xn), ctx.v_Np(xn));
  U_p = element_div(ctx.v_NUp(xn), ctx.v_Nn(xn));

  // Local speed flux -> quantity*Vz
  ctx.v_Np(F) = element_prod(ctx.v_Np(xn), V_z);
  ctx.v_Nn(F) = element_prod(ctx.v_Nn(xn), U_z);
  ctx.v_NVx(F) = element_prod(ctx.v_NVx(xn), V_z);
  ctx.v_NVz(F) = element_prod(ctx.v_NVz(xn), V_z);
  ctx.v_NVp(F) = element_prod(ctx.v_NVp(xn), V_z);
  ctx.v_NUx(F) =  element_prod(ctx.v_NUx(xn), U_z);
  ctx.v_NUz(F) = element_prod(ctx.v_NUz(xn), U_z);
  ctx.v_NUp(F) = element_prod(ctx.v_NUp(xn), U_z);
  ctx.v_Ep(F) = element_prod(ctx.v_Ep(xn), V_z);
  ctx.v_En(F) = element_prod(ctx.v_En(xn), U_z);

  // Induction VB-BV
  ctx.v_Bx(F) = element_prod(V_z, ctx.v_Bx(xn)) - element_prod(ctx.v_Bz(xn), V_x);
  ctx.v_Bp(F) = element_prod(V_z, ctx.v_Bp(xn)) - element_prod(ctx.v_Bz(xn), V_p);

  // General Lagrange Multiplier
  ctx.v_Bz(F) = ctx.v_GLM(xn);

  // Thermal pressure
  vcl_mat Pp = comfi::routines::pressure_p(xn, ctx);
  ctx.v_NVz(F) += Pp;
  vcl_mat Pn = comfi::routines::pressure_n(xn, ctx);
  ctx.v_NUz(F) += Pn;

  // Magnetic pressure
  ctx.v_NVz(F) += 0.5*(element_prod(ctx.v_Bx(xn), ctx.v_Bx(xn))
                       +element_prod(ctx.v_Bz(xn), ctx.v_Bz(xn))
                       +element_prod(ctx.v_Bp(xn), ctx.v_Bp(xn)));
  ctx.v_NVz(F) -= element_prod(ctx.v_Bz(xn), ctx.v_Bz(xn));
  ctx.v_NVx(F) -= element_prod(ctx.v_Bz(xn), ctx.v_Bx(xn));
  ctx.v_NVp(F) -= element_prod(ctx.v_Bz(xn), ctx.v_Bp(xn));

  // Energy flux
  ctx.v_Ep(F) += element_prod(Pp, V_z);
  ctx.v_En(F) += element_prod(Pn, U_z);

  // Flux part of GLM
  const double ch = ds/ctx.dt();
  ctx.v_GLM(F) = ch*ch*ctx.v_Bz(xn);

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

vcl_mat comfi::routines::pressure_n(const vcl_mat &xn, comfi::types::Context &ctx) {
  vcl_mat Pn(xn.size1(), 1);
  Pn = element_prod(viennacl::project(xn, ctx.r_grid(), ctx.r_NUx()), viennacl::project(xn, ctx.r_grid(), ctx.r_NUx()));
  Pn = Pn + element_prod(viennacl::project(xn, ctx.r_grid(), ctx.r_NUz()), viennacl::project(xn, ctx.r_grid(), ctx.r_NUz()));
  Pn = Pn + element_prod(viennacl::project(xn, ctx.r_grid(), ctx.r_NUp()), viennacl::project(xn, ctx.r_grid(), ctx.r_NUp()));
  Pn = 0.5*element_div(Pn, viennacl::project(xn, ctx.r_grid(), ctx.r_Nn()));
  Pn  = element_fabs((gammamono-1.0)*(viennacl::project(xn, ctx.r_grid(), ctx.r_En()) - Pn));
  return Pn;
}

vcl_mat comfi::routines::sound_speed_neutral_mat(const vcl_mat &xn, comfi::types::Context &ctx) {
  vcl_mat k_e = element_prod(ctx.v_NUx(xn), ctx.v_NUx(xn));
  k_e = k_e + element_prod(ctx.v_NUz(xn), ctx.v_NUz(xn));
  k_e = k_e + element_prod(ctx.v_NUp(xn), ctx.v_NUp(xn));
  k_e = 0.5*element_div(k_e, ctx.v_Nn(xn));
  const vcl_mat En = element_fabs(ctx.v_En(xn));
  const vcl_mat Pn  = element_fabs((gammamono-1.0)*(En - k_e));
  return element_sqrt(element_div(gammamono*Pn, ctx.v_Nn(xn)));
}

vcl_vec comfi::routines::sound_speed_neutral(const vcl_mat &xn, const comfi::types::Context &ctx) {
  const vcl_vec NUx = viennacl::column(xn, Ux);
  const vcl_vec NUp = viennacl::column(xn, Up);
  const vcl_vec NUz = viennacl::column(xn, Uz);
  vcl_vec k_e = element_prod(NUx, NUx);
  k_e = k_e + element_prod(NUz, NUz);
  k_e = k_e + element_prod(NUp, NUp);
  const vcl_vec Nn = viennacl::column(xn, n_n);
  k_e = 0.5*element_div(k_e, Nn);
  const vcl_vec En = viennacl::column(xn, E_n);
  const vcl_vec Pn  = element_fabs((gammamono-1.0)*(En - k_e));
  return element_sqrt(element_div(gammamono*Pn, Nn));
}

vcl_mat comfi::routines::pressure_p(const vcl_mat &xn, comfi::types::Context &ctx)
{
  using namespace viennacl::linalg;
  // Calculate pressures by total energy
  vcl_mat Pp(xn.size1(), 1);
  Pp = element_prod(viennacl::project(xn, ctx.r_grid(), ctx.r_NVx()), viennacl::project(xn, ctx.r_grid(), ctx.r_NVx()));
  Pp = Pp + element_prod(viennacl::project(xn, ctx.r_grid(), ctx.r_NVz()), viennacl::project(xn, ctx.r_grid(), ctx.r_NVz()));
  Pp = Pp + element_prod(viennacl::project(xn, ctx.r_grid(), ctx.r_NVp()), viennacl::project(xn, ctx.r_grid(), ctx.r_NVp()));
  Pp = 0.5*element_div(Pp, viennacl::project(xn, ctx.r_grid(), ctx.r_Np()));

  Pp = Pp + 0.5*element_prod(viennacl::project(xn, ctx.r_grid(), ctx.r_Bx()), viennacl::project(xn, ctx.r_grid(), ctx.r_Bx()));
  Pp = Pp + 0.5*element_prod(viennacl::project(xn, ctx.r_grid(), ctx.r_Bz()), viennacl::project(xn, ctx.r_grid(), ctx.r_Bz()));
  Pp = Pp + 0.5*element_prod(viennacl::project(xn, ctx.r_grid(), ctx.r_Bp()), viennacl::project(xn, ctx.r_grid(), ctx.r_Bp()));

  Pp = element_fabs((gammamono-1.0)*(viennacl::project(xn, ctx.r_grid(), ctx.r_Ep()) - Pp));

  return Pp;
}

vcl_mat comfi::routines::fast_speed_x_mat(const vcl_mat &xn, comfi::types::Context &ctx) {
  using namespace viennacl::linalg;
  // Calculate pressures by total energy
  vcl_mat k_e = element_prod(ctx.v_NVx(xn), ctx.v_NVx(xn));
  k_e = k_e + element_prod(ctx.v_NVz(xn), ctx.v_NVz(xn));
  k_e = k_e + element_prod(ctx.v_NVp(xn), ctx.v_NVp(xn));
  k_e = 0.5*element_div(k_e, ctx.v_Np(xn));

  vcl_mat b_e = 0.5*element_prod(ctx.v_Bx(xn), ctx.v_Bx(xn));
  b_e = b_e + 0.5*element_prod(ctx.v_Bz(xn), ctx.v_Bz(xn));
  b_e = b_e + 0.5*element_prod(ctx.v_Bp(xn), ctx.v_Bp(xn));

  const vcl_mat Ep = element_fabs(ctx.v_Ep(xn));
  const vcl_mat Pp  = element_fabs((gammamono-1.0)*(Ep - k_e - b_e));

  const vcl_mat cps2 = gammamono*(element_div(Pp, ctx.v_Np(xn)));
  const vcl_mat cps = element_sqrt(cps2);
  const vcl_mat ca2 = element_div(2.0*b_e, ctx.v_Np(xn));
  const vcl_mat cax = element_div(ctx.v_Bx(xn), element_sqrt(ctx.v_Np(xn)));
  const vcl_mat cpsca = element_prod(cps, cax);
  const vcl_mat cpsca2 = element_prod(cpsca, cpsca);

  const vcl_mat cp = 0.5*(element_sqrt(2.0*(cps2 + ca2 + element_sqrt(element_prod(cps2+ca2,cps2+ca2)-4.0*cpsca2))));

  return cp;
}

vcl_mat comfi::routines::fast_speed_z_mat(const vcl_mat &xn, comfi::types::Context &ctx) {
  using namespace viennacl::linalg;
  // Calculate pressures by total energy
  vcl_mat k_e = element_prod(ctx.v_NVx(xn), ctx.v_NVx(xn));
  k_e = k_e + element_prod(ctx.v_NVz(xn), ctx.v_NVz(xn));
  k_e = k_e + element_prod(ctx.v_NVp(xn), ctx.v_NVp(xn));
  k_e = 0.5*element_div(k_e, ctx.v_Np(xn));

  vcl_mat b_e = 0.5*element_prod(ctx.v_Bx(xn), ctx.v_Bx(xn));
  b_e = b_e + 0.5*element_prod(ctx.v_Bz(xn), ctx.v_Bz(xn));
  b_e = b_e + 0.5*element_prod(ctx.v_Bp(xn), ctx.v_Bp(xn));

  const vcl_mat Ep = element_fabs(ctx.v_Ep(xn));
  const vcl_mat Pp  = element_fabs((gammamono-1.0)*(Ep - k_e - b_e));

  const vcl_mat cps2 = gammamono*(element_div(Pp, ctx.v_Np(xn)));
  const vcl_mat cps = element_sqrt(cps2);
  const vcl_mat ca2 = element_div(2.0*b_e, ctx.v_Np(xn));
  const vcl_mat caz = element_div(ctx.v_Bz(xn), element_sqrt(ctx.v_Np(xn)));
  const vcl_mat cpsca = element_prod(cps, caz);
  const vcl_mat cpsca2 = element_prod(cpsca, cpsca);

  const vcl_mat cp = 0.5*(element_sqrt(2.0*(cps2 + ca2 + element_sqrt(element_prod(cps2+ca2,cps2+ca2)-4.0*cpsca2))));

  return cp;
}

vcl_vec comfi::routines::fast_speed_z(const vcl_mat &xn, comfi::types::Context &ctx)
{
  using namespace viennacl::linalg;
  // Calculate pressures by total energy
  const vcl_vec Np = viennacl::column(xn, n_p);
  const vcl_vec NVx = viennacl::column(xn, Vx);
  const vcl_vec NVp = viennacl::column(xn, Vp);
  const vcl_vec NVz = viennacl::column(xn, Vz);
  vcl_vec k_e = element_prod(NVx, NVx);
  k_e = k_e + element_prod(NVz, NVz);
  k_e = k_e + element_prod(NVp, NVp);
  k_e = 0.5*element_div(k_e, Np);

  const vcl_vec v_Bx = viennacl::column(xn, Bx);
  const vcl_vec v_Bz = viennacl::column(xn, Bz);
  const vcl_vec v_Bp = viennacl::column(xn, Bp);
  vcl_vec b_e = 0.5*element_prod(v_Bx, v_Bx);
  b_e = b_e + 0.5*element_prod(v_Bz, v_Bz);
  b_e = b_e + 0.5*element_prod(v_Bp, v_Bp);

  const vcl_vec Ep = viennacl::column(xn, E_p);
  const vcl_vec Pp  = element_fabs((gammamono-1.0)*(Ep - k_e - b_e));

  const vcl_vec cps2 = gammamono*(element_div(Pp, Np));
  const vcl_vec cps = element_sqrt(cps2);
  const vcl_vec ca2 = element_div(2.0*b_e, Np);
  const vcl_vec caz = element_div(v_Bz, element_sqrt(Np));
  const vcl_vec cpsca = element_prod(cps, caz);
  const vcl_vec cpsca2 = element_prod(cpsca, cpsca);

  const vcl_vec cp = 0.5*(element_sqrt(2.0*(cps2 + ca2 + element_sqrt(element_prod(cps2+ca2,cps2+ca2)-4.0*cpsca2))));

  return cp;
}

vcl_vec comfi::routines::fast_speed_x(const vcl_mat &xn, comfi::types::Context &ctx)
{
  using namespace viennacl::linalg;
  // Calculate pressures by total energy
  const vcl_vec Np = viennacl::column(xn, n_p);
  const vcl_vec NVx = viennacl::column(xn, Vx);
  const vcl_vec NVp = viennacl::column(xn, Vp);
  const vcl_vec NVz = viennacl::column(xn, Vz);
  vcl_vec k_e = element_prod(NVx, NVx);
  k_e = k_e + element_prod(NVz, NVz);
  k_e = k_e + element_prod(NVp, NVp);
  k_e = 0.5*element_div(k_e, Np);

  const vcl_vec v_Bx = viennacl::column(xn, Bx);
  const vcl_vec v_Bz = viennacl::column(xn, Bz);
  const vcl_vec v_Bp = viennacl::column(xn, Bp);
  vcl_vec b_e = 0.5*element_prod(v_Bx, v_Bx);
  b_e = b_e + 0.5*element_prod(v_Bz, v_Bz);
  b_e = b_e + 0.5*element_prod(v_Bp, v_Bp);

  const vcl_vec Ep = viennacl::column(xn, E_p);
  const vcl_vec Pp  = element_fabs((gammamono-1.0)*(Ep - k_e - b_e));

  const vcl_vec cps2 = gammamono*(element_div(Pp, Np));
  const vcl_vec cps = element_sqrt(cps2);
  const vcl_vec ca2 = element_div(2.0*b_e, Np);
  const vcl_vec cax = element_div(v_Bx, element_sqrt(Np));
  const vcl_vec cpsca = element_prod(cps, cax);
  const vcl_vec cpsca2 = element_prod(cpsca, cpsca);

  const vcl_vec cp = 0.5*(element_sqrt(2.0*(cps2 + ca2 + element_sqrt(element_prod(cps2+ca2,cps2+ca2)-4.0*cpsca2))));

  return cp;
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
  // Ospre
  const vcl_mat r2 = element_prod(r, r);
  //return 1.5*element_div(r2+r, r2+r+ones);
  // Van Albada
  return element_div(r2+r, r2+ones);
  // Van Leer
  //const vcl_mat absr = element_fabs(r);
  //return element_div(r+absr, ones+absr);
}

void comfi::routines::bottombc_shock_tube(vcl_mat &Lxn, vcl_mat &Rxn, comfi::types::Context &ctx) {
  uint ij = inds(0, 0, ctx);

  Rxn(ij, n_n) = 0.125;
  Rxn(ij, n_p) = 0.125;
  Rxn(ij, E_p) = 0.1/(gammamono-1.0);
  Rxn(ij, E_n) = 0.1/(gammamono-1.0);
  Rxn(ij, Ux) = 0.0;
  Rxn(ij, Uz) = 0.0;
  Rxn(ij, Up) = 0.0;
  Rxn(ij, Vx) = 0.0;
  Rxn(ij, Vz) = 0.0;
  Rxn(ij, Vp) = 0.0;

  Lxn(ij, n_n) = 0.125;
  Lxn(ij, n_p) = 0.125;
  Lxn(ij, E_p) = 0.1/(gammamono-1.0);
  Lxn(ij, E_n) = 0.1/(gammamono-1.0);
  Lxn(ij, Ux) = 0.0;
  Lxn(ij, Uz) = 0.0;
  Lxn(ij, Up) = 0.0;
  Lxn(ij, Vx) = 0.0;
  Lxn(ij, Vz) = 0.0;
  Lxn(ij, Vp) = 0.0;
}

void comfi::routines::topbc_shock_tube(vcl_mat &Lxn, vcl_mat &Rxn, comfi::types::Context &ctx) {
  uint ij = inds(0, ctx.nz()-1, ctx);

  Rxn(ij, n_n) = 1.0;
  Rxn(ij, n_p) = 1.0;
  Rxn(ij, E_p) = 1.0/(gammamono-1.0);
  Rxn(ij, E_n) = 1.0/(gammamono-1.0);
  Rxn(ij, Ux) = 0.0;
  Rxn(ij, Uz) = 0.0;
  Rxn(ij, Up) = 0.0;
  Rxn(ij, Vx) = 0.0;
  Rxn(ij, Vz) = 0.0;
  Rxn(ij, Vp) = 0.0;

  Lxn(ij, n_n) = 1.0;
  Lxn(ij, n_p) = 1.0;
  Lxn(ij, E_p) = 1.0/(gammamono-1.0);
  Lxn(ij, E_n) = 1.0/(gammamono-1.0);
  Lxn(ij, Ux) = 0.0;
  Lxn(ij, Uz) = 0.0;
  Lxn(ij, Up) = 0.0;
  Lxn(ij, Vx) = 0.0;
  Lxn(ij, Vz) = 0.0;
  Lxn(ij, Vp) = 0.0;
}
