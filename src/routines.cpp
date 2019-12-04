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

using namespace viennacl::linalg;
using namespace arma;


/* sp_mat comfi::routines::computeRi(const vcl_vec &xn_vcl, const comfi::types::Operators &op) */
/*{*/
/*  vec xn(num_of_elem);*/
/*  viennacl::fast_copy(xn_vcl, xn);*/

/*  const uint nnzp = 12;*/
/*  umat  Avi = zeros<umat>(nnzp, num_of_grid);*/
/*  umat  Avj = zeros<umat>(nnzp, num_of_grid);*/
/*  mat   Avv = zeros<mat> (nnzp, num_of_grid);*/

/*  #pragma omp parallel for schedule(static)*/
/*  for (uint index=0; index<num_of_grid; index++)*/
/*  {*/
/*    // indexing*/
/*    const unsigned int  i=(index)%nx; // find i and c++ index ii*/
/*    const unsigned int  j=(index)/nx; //find j and c++ index jj*/

/*    const int           ij=ind(i, j);*/
/*    int                 ip1j=ind(i+1, j);*/
/*    int                 im1j=ind(i-1, j);*/
/*    int                 ijp1=ind(i, j+1);*/
/*    int                 ijm1=ind(i, j-1);*/

/*    //const double nuin   = mhdsim::sol::nu_in(Nnij, 0.5*(Tpij+Tnij));*/
/*    //const double nuni   = mhdsim::sol::nu_in(Nnij, 0.5*(Tpij+Tnij))*Npij/Nnij;*/
/*    const double nuni   = collisionrate;*/
/*    const double nuin   = nuni*Nnij/Npij;*/

/*    //const double resij = mhdsim::sol::resistivity(Npij, Nnij, Tpij, Tnij);*/

/*    //const double irate = mhdsim::sol::ionization_coeff(Tnij);*/
/*    //const double rrate = mhdsim::sol::recomb_coeff(Tpij);*/

/*    // collect*/
/*    Avi.col(index) = vi;*/
/*    Avj.col(index) = vj;*/
/*    Avv.col(index) = vv;*/
/*  }*/

/*  return comfi::util::syncSpMat(Avi, Avj, Avv);*/
/*}*/

vcl_mat comfi::routines::build_eig_matrix_z(const vcl_mat &xn,
                                            comfi::types::Context &ctx) {
  vcl_mat eig_matrix(ctx.num_of_grid(), ctx.num_of_eq);
  const vcl_mat p_fast = comfi::routines::fast_speed_z(xn, ctx);
  const vcl_mat GLM_eig = viennacl::scalar_matrix<double>(p_fast.size1(), p_fast.size2(), ctx.c_h()); 
  const vcl_mat V_z = element_fabs(viennacl::linalg::element_div(ctx.v_NVz(xn), ctx.v_Np(xn)));
  const vcl_mat U_z = element_fabs(viennacl::linalg::element_div(ctx.v_NUz(xn), ctx.v_Nn(xn)));
  const vcl_mat c_a = viennacl::linalg::element_div(ctx.v_Bz(xn), viennacl::linalg::element_sqrt(ctx.v_Np(xn)));
  const vcl_mat c_s = comfi::routines::sound_speed_p(xn, ctx);
  const vcl_mat c_sn = comfi::routines::sound_speed_n(xn, ctx);

  //ctx.v_Np(eig_matrix) = -GLM_eig;
  ctx.v_Np(eig_matrix) = V_z+p_fast;
  ctx.v_NVx(eig_matrix) = V_z+p_fast;
  ctx.v_NVp(eig_matrix) = V_z+c_a;
  ctx.v_NVz(eig_matrix) = V_z+p_fast;
  ctx.v_Bx(eig_matrix) = V_z;
  ctx.v_Bp(eig_matrix) = V_z+c_s;
  ctx.v_Bz(eig_matrix) = V_z+c_a;
  ctx.v_Ep(eig_matrix) = V_z+p_fast;
  ctx.v_Nn(eig_matrix) = U_z+c_sn;
  ctx.v_NUx(eig_matrix) = U_z+c_sn;
  ctx.v_NUp(eig_matrix) = U_z+c_sn;
  ctx.v_NUz(eig_matrix) = U_z+c_sn;
  ctx.v_En(eig_matrix) = U_z+c_sn;
  ctx.v_GLM(eig_matrix) = GLM_eig;

  return eig_matrix;
}

vcl_mat comfi::routines::build_eig_matrix_x(const vcl_mat &xn,
                                            comfi::types::Context &ctx) {
  vcl_mat eig_matrix(ctx.num_of_grid(), ctx.num_of_eq);
  const vcl_mat p_fast = comfi::routines::fast_speed_x(xn, ctx);
  const vcl_mat GLM_eig = viennacl::scalar_matrix<double>(p_fast.size1(), p_fast.size2(), ctx.c_h()); 
  const vcl_mat V_x = viennacl::linalg::element_div(ctx.v_NVx(xn), ctx.v_Np(xn));
  const vcl_mat U_x = viennacl::linalg::element_div(ctx.v_NUx(xn), ctx.v_Nn(xn));
  const vcl_mat c_a = viennacl::linalg::element_div(ctx.v_Bx(xn), viennacl::linalg::element_sqrt(ctx.v_Np(xn)));
  const vcl_mat c_s = comfi::routines::sound_speed_p(xn, ctx);
  const vcl_mat c_sn = comfi::routines::sound_speed_n(xn, ctx);

  //ctx.v_Np(eig_matrix) = -GLM_eig;
  ctx.v_Np(eig_matrix) = V_x+p_fast;
  ctx.v_NVx(eig_matrix) = V_x-p_fast;
  ctx.v_NVp(eig_matrix) = V_x-c_a;
  ctx.v_NVz(eig_matrix) = V_x-c_s;
  ctx.v_Bx(eig_matrix) = V_x;
  ctx.v_Bp(eig_matrix) = V_x+c_s;
  ctx.v_Bz(eig_matrix) = V_x+c_a;
  ctx.v_Ep(eig_matrix) = V_x+p_fast;
  ctx.v_Nn(eig_matrix) = U_x+c_sn;
  ctx.v_NUx(eig_matrix) = U_x-c_sn;
  ctx.v_NUp(eig_matrix) = U_x;
  ctx.v_NUz(eig_matrix) = U_x-c_sn;
  ctx.v_En(eig_matrix) = U_x+c_sn;
  ctx.v_GLM(eig_matrix) = GLM_eig;

  return eig_matrix;
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

  static const vcl_mat eps = viennacl::scalar_matrix<double>(xn.size1(), xn.size2(), 1.e-100);
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

  /* // Fast mode speed eigenvalues */
  /* vcl_mat Leig_iph_p = comfi::routines::fast_speed_x_mat(Lxn_iph, ctx); */
  /* vcl_mat Reig_iph_p = comfi::routines::fast_speed_x_mat(Rxn_iph, ctx); */
  /* vcl_mat Leig_jph_p = comfi::routines::fast_speed_z_mat(Lxn_jph, ctx); */
  /* vcl_mat Reig_jph_p = comfi::routines::fast_speed_z_mat(Rxn_jph, ctx); */
  /* vcl_mat Leig_imh_p = comfi::routines::fast_speed_x_mat(Lxn_imh, ctx); */
  /* vcl_mat Reig_imh_p = comfi::routines::fast_speed_x_mat(Rxn_imh, ctx); */
  /* vcl_mat Leig_jmh_p = comfi::routines::fast_speed_z_mat(Lxn_jmh, ctx); */
  /* vcl_mat Reig_jmh_p = comfi::routines::fast_speed_z_mat(Rxn_jmh, ctx); */

  /* viennacl::ocl::program & eig_prog  = viennacl::ocl::current_context().get_program("element_max"); */
  /* viennacl::ocl::kernel  & element_max = eig_prog.get_kernel("element_max"); */

  /* vcl_mat a_imh_p(Leig_iph_p.size1(), Leig_iph_p.size2()); */
  /* viennacl::ocl::enqueue(element_max(Leig_imh_p, Reig_imh_p, */
  /*                                    a_imh_p, */
  /*                                    cl_uint(Leig_imh_p.size1()))); */
  /* vcl_mat a_iph_p(Leig_iph_p.size1(), Leig_iph_p.size2()); */
  /* viennacl::ocl::enqueue(element_max(Leig_iph_p, Reig_iph_p, */
  /*                                    a_iph_p, */
  /*                                    cl_uint(Leig_iph_p.size1()))); */
  /* vcl_mat a_jmh_p(Leig_iph_p.size1(), Leig_iph_p.size2()); */
  /* viennacl::ocl::enqueue(element_max(Leig_jmh_p, Reig_jmh_p, */
  /*                                    a_jmh_p, */
  /*                                    cl_uint(Leig_jmh_p.size1()))); */
  /* vcl_mat a_jph_p(Leig_iph_p.size1(), Leig_iph_p.size2()); */
  /* viennacl::ocl::enqueue(element_max(Leig_jph_p, Reig_jph_p, */
  /*                                    a_jph_p, */
  /*                                    cl_uint(Leig_jph_p.size1()))); */
  /* Leig_iph_p = element_fabs(element_div(ctx.v_NVx(Lxn_iph), ctx.v_Np(Lxn_iph))); */
  /* viennacl::ocl::enqueue(element_max(Leig_iph_p, a_iph_p, */
  /*                                    a_iph_p, */
  /*                                    cl_uint(Leig_iph_p.size1()))); */

  /* Reig_iph_p = element_fabs(element_div(ctx.v_NVx(Rxn_iph), ctx.v_Np(Rxn_iph))); */
  /* viennacl::ocl::enqueue(element_max(Reig_iph_p, a_iph_p, */
  /*                                    a_iph_p, */
  /*                                    cl_uint(Reig_iph_p.size1()))); */
  /* Leig_jph_p = element_fabs(element_div(ctx.v_NVz(Lxn_jph), ctx.v_Np(Lxn_jph))); */
  /* viennacl::ocl::enqueue(element_max(Leig_jph_p, a_jph_p, */
  /*                                    a_jph_p, */
  /*                                    cl_uint(Leig_jph_p.size1()))); */
  /* Reig_jph_p = element_fabs(element_div(ctx.v_NVz(Rxn_jph), ctx.v_Np(Rxn_jph))); */
  /* viennacl::ocl::enqueue(element_max(Reig_jph_p, a_jph_p, */
  /*                                    a_jph_p, */
  /*                                    cl_uint(Reig_jph_p.size1()))); */
  /* Leig_imh_p = element_fabs(element_div(ctx.v_NVx(Lxn_imh), ctx.v_Np(Lxn_imh))); */
  /* viennacl::ocl::enqueue(element_max(Leig_imh_p, a_imh_p, */
  /*                                    a_imh_p, */
  /*                                    cl_uint(Leig_imh_p.size1()))); */
  /* Reig_imh_p = element_fabs(element_div(ctx.v_NVx(Rxn_imh), ctx.v_Np(Rxn_imh))); */
  /* viennacl::ocl::enqueue(element_max(Reig_imh_p, a_imh_p, */
  /*                                    a_imh_p, */
  /*                                    cl_uint(Reig_imh_p.size1()))); */
  /* Leig_jmh_p = element_fabs(element_div(ctx.v_NVz(Lxn_jmh), ctx.v_Np(Lxn_jmh))); */
  /* viennacl::ocl::enqueue(element_max(Leig_jmh_p, a_jmh_p, */
  /*                                    a_jmh_p, */
  /*                                    cl_uint(Leig_jmh_p.size1()))); */
  /* Reig_jmh_p = element_fabs(element_div(ctx.v_NVz(Rxn_jmh), ctx.v_Np(Rxn_jmh))); */
  /* viennacl::ocl::enqueue(element_max(Reig_jmh_p, a_jmh_p, */
  /*                                    a_jmh_p, */
  /*                                    cl_uint(Reig_jmh_p.size1()))); */

  /* vcl_mat Leig_iph_n = comfi::routines::sound_speed_neutral_mat(Lxn_iph, ctx); */
  /* vcl_mat Reig_iph_n = comfi::routines::sound_speed_neutral_mat(Rxn_iph, ctx); */
  /* vcl_mat Leig_jph_n = comfi::routines::sound_speed_neutral_mat(Lxn_jph, ctx); */
  /* vcl_mat Reig_jph_n = comfi::routines::sound_speed_neutral_mat(Rxn_jph, ctx); */
  /* vcl_mat Leig_imh_n = comfi::routines::sound_speed_neutral_mat(Lxn_imh, ctx); */
  /* vcl_mat Reig_imh_n = comfi::routines::sound_speed_neutral_mat(Rxn_imh, ctx); */
  /* vcl_mat Leig_jmh_n = comfi::routines::sound_speed_neutral_mat(Lxn_jmh, ctx); */
  /* vcl_mat Reig_jmh_n = comfi::routines::sound_speed_neutral_mat(Rxn_jmh, ctx); */

  /* vcl_mat a_imh_n(Leig_iph_n.size1(), Leig_iph_n.size2()); */
  /* if (ctx.bc_left != comfi::types::DIMENSIONLESS) { */
  /*   viennacl::ocl::enqueue(element_max(Leig_imh_n, Reig_imh_n, */
  /*                                      a_imh_n, */
  /*                                      cl_uint(Leig_imh_n.size1()))); */
  /* } */
  /* vcl_mat a_iph_n(Leig_iph_n.size1(), Leig_iph_n.size2()); */
  /* if (ctx.bc_right != comfi::types::DIMENSIONLESS) { */
  /*   viennacl::ocl::enqueue(element_max(Leig_iph_n, Reig_iph_n, */
  /*                                      a_iph_n, */
  /*                                      cl_uint(Leig_iph_n.size1()))); */
  /* } */
  /* vcl_mat a_jmh_n(Leig_iph_n.size1(), Leig_iph_n.size2()); */
  /* if (ctx.bc_down != comfi::types::DIMENSIONLESS) { */
  /*   viennacl::ocl::enqueue(element_max(Leig_jmh_n, Reig_jmh_n, */
  /*                                      a_jmh_n, */
  /*                                      cl_uint(Leig_jmh_n.size1()))); */
  /* } */
  /* vcl_mat a_jph_n(Leig_iph_n.size1(), Leig_iph_n.size2()); */
  /* if (ctx.bc_up != comfi::types::DIMENSIONLESS) { */
  /*   viennacl::ocl::enqueue(element_max(Leig_jph_n, Reig_jph_n, */
  /*                                      a_jph_n, */
  /*                                      cl_uint(Leig_jph_n.size1()))); */
  /* } */

  /* Reig_jph_n = element_fabs(element_div(ctx.v_NUz(Rxn_jph), ctx.v_Nn(Rxn_jph))); */
  /* viennacl::ocl::enqueue(element_max(Reig_jph_n, a_jph_n, */
  /*                                    a_jph_n, */
  /*                                    cl_uint(Reig_jph_n.size1()))); */
  /* Reig_jmh_n = element_fabs(element_div(ctx.v_NUz(Rxn_jmh), ctx.v_Nn(Rxn_jmh))); */
  /* viennacl::ocl::enqueue(element_max(Reig_jmh_n, a_jmh_n, */
  /*                                    a_jmh_n, */
  /*                                    cl_uint(Reig_jmh_n.size1()))); */
  /* Reig_imh_n = element_fabs(element_div(ctx.v_NUx(Rxn_imh), ctx.v_Nn(Rxn_imh))); */
  /* viennacl::ocl::enqueue(element_max(Reig_imh_n, a_imh_n, */
  /*                                    a_imh_n, */
  /*                                    cl_uint(Reig_imh_n.size1()))); */
  /* Reig_iph_n = element_fabs(element_div(ctx.v_NUx(Rxn_iph), ctx.v_Nn(Rxn_iph))); */
  /* viennacl::ocl::enqueue(element_max(Reig_iph_n, a_iph_n, */
  /*                                    a_iph_n, */
  /*                                    cl_uint(Reig_iph_n.size1()))); */
  /* Leig_jph_n = element_fabs(element_div(ctx.v_NUz(Lxn_jph), ctx.v_Nn(Lxn_jph))); */
  /* viennacl::ocl::enqueue(element_max(Leig_jph_n, a_jph_n, */
  /*                                    a_jph_n, */
  /*                                    cl_uint(Leig_jph_n.size1()))); */
  /* Leig_jmh_n = element_fabs(element_div(ctx.v_NUz(Lxn_jmh), ctx.v_Nn(Lxn_jmh))); */
  /* viennacl::ocl::enqueue(element_max(Leig_jmh_n, a_jmh_n, */
  /*                                    a_jmh_n, */
  /*                                    cl_uint(Leig_jmh_n.size1()))); */
  /* Leig_imh_n = element_fabs(element_div(ctx.v_NUx(Lxn_imh), ctx.v_Nn(Lxn_imh))); */
  /* viennacl::ocl::enqueue(element_max(Leig_imh_n, a_imh_n, */
  /*                                    a_imh_n, */
  /*                                    cl_uint(Leig_imh_n.size1()))); */
  /* Leig_iph_n = element_fabs(element_div(ctx.v_NUx(Lxn_iph), ctx.v_Nn(Lxn_iph))); */
  /* viennacl::ocl::enqueue(element_max(Leig_iph_n, a_iph_n, */
  /*                                    a_iph_n, */
  /*                                    cl_uint(Leig_iph_n.size1()))); */

  const vcl_mat a_imh = element_fabs(build_eig_matrix_x(0.5*(Lxn_imh+Rxn_imh), ctx));
  const vcl_mat a_iph = element_fabs(build_eig_matrix_x(0.5*(Lxn_iph+Rxn_iph), ctx));
  const vcl_mat a_jmh = element_fabs(build_eig_matrix_z(0.5*(Lxn_jmh+Rxn_jmh), ctx));
  const vcl_mat a_jph = element_fabs(build_eig_matrix_z(0.5*(Lxn_jph+Rxn_jph), ctx));

  // LAX-FRIEDRICHS FLUX
  const vcl_mat Fximh = 0.5*(comfi::routines::Fx(Lxn_imh, xn, ctx)+comfi::routines::Fx(Rxn_imh, xn, ctx))
                             -element_prod(a_imh, (Rxn_imh-Lxn_imh));
  const vcl_mat Fxiph = 0.5*(comfi::routines::Fx(Lxn_iph, xn, ctx)+comfi::routines::Fx(Rxn_iph, xn, ctx))
                             -element_prod(a_iph, (Rxn_iph-Lxn_iph));
  const vcl_mat Fzjmh = 0.5*(comfi::routines::Fz(Lxn_jmh, xn, ctx)+comfi::routines::Fz(Rxn_jmh, xn, ctx))
                             -element_prod(a_jmh, (Rxn_jmh-Lxn_jmh));
  const vcl_mat Fzjph = 0.5*(comfi::routines::Fz(Lxn_jph, xn, ctx)+comfi::routines::Fz(Rxn_jph, xn, ctx))
                             -element_prod(a_jph, (Rxn_jph-Lxn_jph));

  return -1.0*(Fxiph-Fximh)/ctx.dx
         -1.0*(Fzjph-Fzjmh)/ctx.dz;
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

vcl_mat comfi::routines::computeRHS_RK4(const vcl_mat &xn, comfi::types::Context &ctx)
{
  const double dt = ctx.dt();
  // RK-4
  const vcl_mat k1 = Re_MUSCL(xn, ctx)*dt; //return xn+k1;
  //const vcl_mat k2 = Re_MUSCL(xn+0.5*k1,t+0.5*dt,op,bg)*dt;
  const vcl_mat k2 = Re_MUSCL(xn+0.5*k1, ctx)*dt;
  //const vcl_mat k3 = Re_MUSCL(xn+0.5*k2,t+0.5*dt,op,bg)*dt;
  const vcl_mat k3 = Re_MUSCL(xn+0.5*k2, ctx)*dt;
  //const vcl_mat k4 = Re_MUSCL(xn+k3,t+dt,op,bg)*dt;
  const vcl_mat k4 = Re_MUSCL(xn+k3, ctx)*dt;

  vcl_mat result = xn + (k1+2.0*k2+2.0*k3+k4)/6.0;

  // GLM exact solution
  ctx.v_GLM(result) *= std::exp(-ctx.alpha_p*ctx.dt()*ctx.c_h()/ctx.ds);

  return result;
}

vcl_mat comfi::routines::computeRHS_Euler(const vcl_mat &xn, comfi::types::Context &ctx)
{
  // Simple Eulerian Steps
  vcl_mat result = xn + comfi::routines::Re_MUSCL(xn, ctx)*ctx.dt();

  // GLM exact solution
  ctx.v_GLM(result) *= std::exp(-ctx.alpha_p*ctx.dt()*ctx.c_h()/ctx.ds);

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

vcl_mat comfi::routines::Fx(const vcl_mat &xn, const vcl_mat &xn_ij, comfi::types::Context &ctx)
{
  vcl_mat F = viennacl::zero_matrix<double>(xn.size1(), xn.size2());

  const vcl_mat V_x = element_div(ctx.v_NVx(xn), ctx.v_Np(xn));
  const vcl_mat U_x = element_div(ctx.v_NUx(xn), ctx.v_Nn(xn));
  const vcl_mat V_z = element_div(ctx.v_NVz(xn), ctx.v_Np(xn));
  const vcl_mat U_z = element_div(ctx.v_NUz(xn), ctx.v_Nn(xn));
  const vcl_mat V_p = element_div(ctx.v_NVp(xn), ctx.v_Np(xn));
  const vcl_mat U_p = element_div(ctx.v_NUp(xn), ctx.v_Nn(xn));

  // Local speed flux -> quantity*Vz
  ctx.v_Np(F) = element_prod(ctx.v_Np(xn), V_x);
  ctx.v_Nn(F) = element_prod(ctx.v_Nn(xn), U_x);
  ctx.v_NVx(F) = element_prod(ctx.v_NVx(xn), V_x);
  ctx.v_NVz(F) = element_prod(ctx.v_NVz(xn), V_x);
  ctx.v_NVp(F) = element_prod(ctx.v_NVp(xn), V_x);
  ctx.v_NUx(F) = element_prod(ctx.v_NUx(xn), U_x);
  ctx.v_NUz(F) = element_prod(ctx.v_NUz(xn), U_x);
  ctx.v_NUp(F) = element_prod(ctx.v_NUp(xn), U_x);
  ctx.v_Ep(F) = element_prod(ctx.v_Ep(xn), V_x);
  ctx.v_En(F) = element_prod(ctx.v_En(xn), U_x);

  // Induction VB-BV
  ctx.v_Bz(F) = element_prod(V_x, ctx.v_Bz(xn)) - element_prod(ctx.v_Bx(xn), V_z);
  ctx.v_Bp(F) = element_prod(V_x, ctx.v_Bp(xn)) - element_prod(ctx.v_Bx(xn), V_p);

  // General Lagrange Multiplier
  ctx.v_Bx(F) = ctx.v_GLM(xn);

  // Thermal pressure
  vcl_mat Pp = comfi::routines::pressure_p(xn, ctx);
  ctx.v_NVx(F) += Pp;
  vcl_mat Pn = comfi::routines::pressure_n(xn, ctx);
  ctx.v_NUx(F) += Pn;

  // Magnetic pressure
  const vcl_mat pmag = 0.5*(element_prod(ctx.v_Bx(xn), ctx.v_Bx(xn))
                            + element_prod(ctx.v_Bz(xn), ctx.v_Bz(xn))
                            + element_prod(ctx.v_Bp(xn), ctx.v_Bp(xn)));
  ctx.v_NVx(F) += pmag;
  ctx.v_NVz(F) -= element_prod(ctx.v_Bz(xn), ctx.v_Bz(xn));
  ctx.v_NVx(F) -= element_prod(ctx.v_Bz(xn), ctx.v_Bx(xn));
  ctx.v_NVp(F) -= element_prod(ctx.v_Bz(xn), ctx.v_Bp(xn));

  // Energy flux
  vcl_mat bdotv = element_prod(V_x, ctx.v_Bx(xn)) + element_prod(V_z, ctx.v_Bz(xn)) + element_prod(V_p, ctx.v_Bp(xn));
  ctx.v_Ep(F) += element_prod(Pp+pmag, V_x) - element_prod(ctx.v_Bx(xn), bdotv);
  ctx.v_En(F) += element_prod(Pn, U_x);

  // Flux part of GLM
  ctx.v_GLM(F) = ctx.c_h()*ctx.c_h()*ctx.v_Bx(xn);

  return F;
}

vcl_mat comfi::routines::Fz(const vcl_mat &xn, const vcl_mat &xn_ij, comfi::types::Context &ctx)
{
  vcl_mat F = viennacl::zero_matrix<double>(xn.size1(), xn.size2());

  const vcl_mat V_x = element_div(ctx.v_NVx(xn), ctx.v_Np(xn));
  const vcl_mat U_x = element_div(ctx.v_NUx(xn), ctx.v_Nn(xn));
  const vcl_mat V_z = element_div(ctx.v_NVz(xn), ctx.v_Np(xn));
  const vcl_mat U_z = element_div(ctx.v_NUz(xn), ctx.v_Nn(xn));
  const vcl_mat V_p = element_div(ctx.v_NVp(xn), ctx.v_Np(xn));
  const vcl_mat U_p = element_div(ctx.v_NUp(xn), ctx.v_Nn(xn));

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
  const vcl_mat pmag = 0.5*(element_prod(ctx.v_Bx(xn), ctx.v_Bx(xn))
                            +element_prod(ctx.v_Bz(xn), ctx.v_Bz(xn))
                            +element_prod(ctx.v_Bp(xn), ctx.v_Bp(xn)));
  ctx.v_NVz(F) += pmag;
  ctx.v_NVz(F) -= element_prod(ctx.v_Bz(xn), ctx.v_Bz(xn));
  ctx.v_NVx(F) -= element_prod(ctx.v_Bz(xn), ctx.v_Bx(xn));
  ctx.v_NVp(F) -= element_prod(ctx.v_Bz(xn), ctx.v_Bp(xn));

  // Energy flux
  vcl_mat bdotv = element_prod(V_x, ctx.v_Bx(xn))
                  + element_prod(V_z, ctx.v_Bz(xn))
                  + element_prod(V_p, ctx.v_Bp(xn));
  ctx.v_Ep(F) += element_prod(Pp+pmag, V_z) - element_prod(ctx.v_Bz(xn), bdotv);
  ctx.v_En(F) += element_prod(Pn, U_z);

  // Flux part of GLM
  ctx.v_GLM(F) = ctx.c_h()*ctx.c_h()*ctx.v_Bz(xn);

  return F;
}

vcl_mat comfi::routines::pressure_n(const vcl_mat &xn, comfi::types::Context &ctx) {
  vcl_mat Pn = ctx.v_NUx(xn);
  Pn = element_prod(Pn, ctx.v_NUx(xn));
  Pn = Pn + element_prod(ctx.v_NUz(xn), ctx.v_NUz(xn));
  Pn = Pn + element_prod(ctx.v_NUp(xn), ctx.v_NUp(xn));
  Pn = 0.5*element_div(Pn, ctx.v_Nn(xn));
  Pn  = element_fabs((ctx.gammamono-1.0)*(ctx.v_En(xn)-Pn));
  return Pn;
}

vcl_mat comfi::routines::sound_speed_p(const vcl_mat &xn, comfi::types::Context &ctx) {
  const vcl_mat Pp = comfi::routines::pressure_p(xn, ctx);
  return element_sqrt(element_div(ctx.gammamono*Pp, ctx.v_Np(xn)));
}

vcl_mat comfi::routines::sound_speed_n(const vcl_mat &xn, comfi::types::Context &ctx) {
  const vcl_mat Pn = comfi::routines::pressure_n(xn, ctx);
  return element_sqrt(element_div(ctx.gammamono*Pn, ctx.v_Nn(xn)));
}

vcl_mat comfi::routines::pressure_p(const vcl_mat &xn, comfi::types::Context &ctx)
{
  using namespace viennacl::linalg;
  // Calculate pressures by total energy
  vcl_mat Pp = ctx.v_NVx(xn);
  Pp = element_prod(Pp, ctx.v_NVx(xn));
  Pp = Pp + element_prod(ctx.v_NVz(xn), ctx.v_NVz(xn));
  Pp = Pp + element_prod(ctx.v_NVp(xn), ctx.v_NVp(xn));
  Pp = 0.5*element_div(Pp, ctx.v_Np(xn));

  Pp = Pp + 0.5*element_prod(ctx.v_Bz(xn), ctx.v_Bz(xn));
  Pp = Pp + 0.5*element_prod(ctx.v_Bx(xn), ctx.v_Bx(xn));
  Pp = Pp + 0.5*element_prod(ctx.v_Bp(xn), ctx.v_Bp(xn));

  Pp = element_fabs((ctx.gammamono-1.0)*(ctx.v_Ep(xn)-Pp));

  return Pp;
}

vcl_mat comfi::routines::fast_speed_x(const vcl_mat &xn, comfi::types::Context &ctx) {
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
  const vcl_mat Pp  = element_fabs((ctx.gammamono-1.0)*(Ep - k_e - b_e));

  const vcl_mat cps2 = ctx.gammamono*(element_div(Pp, ctx.v_Np(xn)));
  const vcl_mat cps = element_sqrt(cps2);
  const vcl_mat ca2 = element_div(2.0*b_e, ctx.v_Np(xn));
  const vcl_mat cax = element_div(ctx.v_Bx(xn), element_sqrt(ctx.v_Np(xn)));
  const vcl_mat cpsca = element_prod(cps, cax);
  const vcl_mat cpsca2 = element_prod(cpsca, cpsca);

  const vcl_mat cp = 0.5*(element_sqrt(2.0*(cps2 + ca2 + element_sqrt(element_prod(cps2+ca2,cps2+ca2)-4.0*cpsca2))));

  return cp;
}

vcl_mat comfi::routines::fast_speed_z(const vcl_mat &xn, comfi::types::Context &ctx) {
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
  const vcl_mat Pp  = element_fabs((ctx.gammamono-1.0)*(Ep - k_e - b_e));

  const vcl_mat cps2 = ctx.gammamono*(element_div(Pp, ctx.v_Np(xn)));
  const vcl_mat cps = element_sqrt(cps2);
  const vcl_mat ca2 = element_div(2.0*b_e, ctx.v_Np(xn));
  const vcl_mat caz = element_div(ctx.v_Bz(xn), element_sqrt(ctx.v_Np(xn)));
  const vcl_mat cpsca = element_prod(cps, caz);
  const vcl_mat cpsca2 = element_prod(cpsca, cpsca);

  const vcl_mat cp = 0.5*(element_sqrt(2.0*(cps2 + ca2 + element_sqrt(element_prod(cps2+ca2,cps2+ca2)-4.0*cpsca2))));

  return cp;
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
  // Ospre
  return 1.5*element_div(r2+r, r2+r+ones);
  // Van Albada
  //return element_div(r2+r, r2+ones);
  // Van Leer
  //const vcl_mat absr = element_fabs(r);
  //return element_div(r+absr, ones+absr);
}

void comfi::routines::bottombc_shock_tube(vcl_mat &Lxn, vcl_mat &Rxn, comfi::types::Context &ctx) {
  uint ij = inds(0, 0, ctx);

  Rxn(ij, ctx.n_n) = 0.125;
  Rxn(ij, ctx.n_p) = 0.125;
  Rxn(ij, ctx.E_p) = 0.1/(ctx.gammamono-1.0);
  Rxn(ij, ctx.E_n) = 0.1/(ctx.gammamono-1.0);
  Rxn(ij, ctx.Ux) = 0.0;
  Rxn(ij, ctx.Uz) = 0.0;
  Rxn(ij, ctx.Up) = 0.0;
  Rxn(ij, ctx.Vx) = 0.0;
  Rxn(ij, ctx.Vz) = 0.0;
  Rxn(ij, ctx.Vp) = 0.0;

  Lxn(ij, ctx.n_n) = 0.125;
  Lxn(ij, ctx.n_p) = 0.125;
  Lxn(ij, ctx.E_p) = 0.1/(ctx.gammamono-1.0);
  Lxn(ij, ctx.E_n) = 0.1/(ctx.gammamono-1.0);
  Lxn(ij, ctx.Ux) = 0.0;
  Lxn(ij, ctx.Uz) = 0.0;
  Lxn(ij, ctx.Up) = 0.0;
  Lxn(ij, ctx.Vx) = 0.0;
  Lxn(ij, ctx.Vz) = 0.0;
  Lxn(ij, ctx.Vp) = 0.0;
}

void comfi::routines::topbc_shock_tube(vcl_mat &Lxn, vcl_mat &Rxn, comfi::types::Context &ctx) {
  uint ij = inds(0, ctx.nz-1, ctx);

  Rxn(ij, ctx.n_n) = 1.0;
  Rxn(ij, ctx.n_p) = 1.0;
  Rxn(ij, ctx.E_p) = 1.0/(ctx.gammamono-1.0);
  Rxn(ij, ctx.E_n) = 1.0/(ctx.gammamono-1.0);
  Rxn(ij, ctx.Ux) = 0.0;
  Rxn(ij, ctx.Uz) = 0.0;
  Rxn(ij, ctx.Up) = 0.0;
  Rxn(ij, ctx.Vx) = 0.0;
  Rxn(ij, ctx.Vz) = 0.0;
  Rxn(ij, ctx.Vp) = 0.0;

  Lxn(ij, ctx.n_n) = 1.0;
  Lxn(ij, ctx.n_p) = 1.0;
  Lxn(ij, ctx.E_p) = 1.0/(ctx.gammamono-1.0);
  Lxn(ij, ctx.E_n) = 1.0/(ctx.gammamono-1.0);
  Lxn(ij, ctx.Ux) = 0.0;
  Lxn(ij, ctx.Uz) = 0.0;
  Lxn(ij, ctx.Up) = 0.0;
  Lxn(ij, ctx.Vx) = 0.0;
  Lxn(ij, ctx.Vz) = 0.0;
  Lxn(ij, ctx.Vp) = 0.0;
}

/*
vim: tabstop=2
vim: shiftwidth=2
vim: smarttab
vim: expandtab
*/
