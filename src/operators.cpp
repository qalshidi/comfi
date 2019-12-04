#include "comfi.hpp"
#include <armadillo>
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/forwards.h"

using namespace arma;

vcl_mat comfi::operators::jm1(const vcl_mat &xn, comfi::types::Context ctx) {
  //make sure the dimension exists
  if ((ctx.bc_down == comfi::types::DIMENSIONLESS) ||
      (ctx.nz == 1)) {
    return xn;
  }
  //shift values
  vcl_mat xn_jm1(xn.size1(), xn.size2());
  viennacl::range eq(0, ctx.num_of_eq);
  static viennacl::range jm1(0, ctx.num_of_grid()-ctx.nx);
  static viennacl::range j(ctx.nx, ctx.num_of_grid());
  viennacl::project(xn_jm1, j, eq) = viennacl::project(xn, jm1, eq);

  //boundary conditions
  if (ctx.bc_down == comfi::types::NEUMANN) {
    static viennacl::range edge(0, ctx.nx);
    viennacl::project(xn_jm1, edge, eq) = viennacl::project(xn, edge, eq);
  }
  if (ctx.bc_down == comfi::types::PERIODIC) {
    static viennacl::range start(ctx.num_of_grid()-ctx.nx, ctx.num_of_grid());
    static viennacl::range edge(0, ctx.nx);
    viennacl::project(xn_jm1, edge, eq) = viennacl::project(xn, start, eq);
  }

  return xn_jm1;
}

vcl_mat comfi::operators::jp1(const vcl_mat &xn, comfi::types::Context ctx) {

  //make sure the dimension exists
  if ((ctx.bc_up == comfi::types::DIMENSIONLESS) ||
      (ctx.nz == 1)) {
    return xn;
  }
  
  //shift values
  vcl_mat xn_jp1(xn.size1(), xn.size2());
  static viennacl::range eq(0, ctx.num_of_eq);
  static viennacl::range jp1(inds(0, 1, ctx), ctx.num_of_grid());
  static viennacl::range j(0, ctx.num_of_grid()-ctx.nx);
  viennacl::project(xn_jp1, j, eq) = viennacl::project(xn, jp1, eq);

  //boundary conditions
  if (ctx.bc_up == comfi::types::NEUMANN) {
    static viennacl::range edge(ctx.num_of_grid()-ctx.nx, ctx.num_of_grid());
    viennacl::project(xn_jp1, edge, eq) = viennacl::project(xn, edge, eq);
  }
  if (ctx.bc_up == comfi::types::PERIODIC) {
    static viennacl::range edge(ctx.num_of_grid()-ctx.nx, ctx.num_of_grid());
    static viennacl::range start(0, ctx.nx);
    viennacl::project(xn_jp1, edge, eq) = viennacl::project(xn, start, eq);
  }

  return xn_jp1;
}

vcl_mat comfi::operators::ip1(const vcl_mat &xn, comfi::types::Context ctx) {

  //make sure the dimension exists
  if ((ctx.bc_right == comfi::types::DIMENSIONLESS) ||
      (ctx.nx == 1)) {
    return xn;
  }

  /* // TODO: Fix viennacl so that this method is working (preferred) */
  /* //shift values */
  /* vcl_mat xn_ip1(xn.size1(), xn.size2()); */
  /* viennacl::range eq(0, xn.size1()); */
  /* viennacl::range ip1[ctx.nz]; */
  /* viennacl::range i[ctx.nz]; */
  /* #pragma omp parallel for schedule(dynamic) */
  /* for(uint j = 0; j < ctx.nz; j++) { */
  /*    ip1[j] = viennacl::range(inds(1, j, ctx), inds(ctx.nx-1, j, ctx)+1); */
  /*    i[j] = viennacl::range(inds(0, j, ctx), inds(ctx.nx-2, j, ctx)+1); */
  /* } */
  /* /1* #pragma omp parallel for *1/ */ 
  /* for(uint j = 0; j < ctx.nz; j++) { */
  /*   viennacl::project(xn_ip1, eq, i[j]) = viennacl::project(xn, eq, ip1[j]); */
  /* } */

  /* //boundary conditions */
  /* if (ctx.bc_right == comfi::types::NEUMANN) { */
  /*   viennacl::slice eq(0, 1, ctx.num_of_eq); */
  /*   viennacl::slice edge(inds(ctx.nx-1, 0, ctx), ctx.nx, ctx.nz); */
  /*   viennacl::project(xn_ip1, edge, eq) = viennacl::project(xn, edge, eq); */
  /* } */
  /* if (ctx.bc_right == comfi::types::PERIODIC) { */
  /*   viennacl::slice eq(0, 1, ctx.num_of_eq); */
  /*   viennacl::slice edge(inds(ctx.nx-1, 0, ctx), ctx.nx, ctx.nz); */
  /*   viennacl::slice start(0, ctx.nx, ctx.nz); */
  /*   viennacl::project(xn_ip1, edge, eq) = viennacl::project(xn, start, eq); */
  /* } */

  static const sp_mat cpu_Pip1 = comfi::operators::buildPip1(ctx);
  static vcl_sp_mat Pip1(ctx.num_of_grid(), ctx.num_of_grid());
  static bool created = false;
  if (!created) { viennacl::copy(cpu_Pip1, Pip1); created = true; }
  vcl_mat xn_ip1 = viennacl::linalg::prod(Pip1, xn);

  return xn_ip1;
}

vcl_mat comfi::operators::im1(const vcl_mat &xn, comfi::types::Context ctx) {
  //make sure the dimension exists
  if ((ctx.bc_left == comfi::types::DIMENSIONLESS) ||
      (ctx.nx == 1)) {
    return xn;
  }
  
  /* // TODO: Make this method work through changing viennacl (preferred) */
  /* //shift values */
  /* vcl_mat xn_im1(xn.size1(), xn.size2()); */
  /* viennacl::range eq(0, xn.size1()); */
  /* viennacl::range im1[ctx.nz]; */
  /* viennacl::range i[ctx.nz]; */
  /* #pragma omp parallel for schedule(dynamic) */
  /* for(uint j = 0; j < ctx.nz; j++) { */
  /*   im1[j] = viennacl::range(inds(0, j, ctx), inds(ctx.nx()-2, j, ctx)+1); */
  /*   i[j] = viennacl::range(inds(1, j, ctx), inds(ctx.nx()-1, j, ctx)+1); */
  /* } */
  /* for(uint j = 0; j < ctx.nz(); j++) { */
  /*   viennacl::project(xn_im1, eq, i[j]) = viennacl::project(xn, eq, im1[j]); */
  /* } */

  /* //boundary conditions */
  /* if (ctx.bc_left == comfi::types::NEUMANN) { */
  /*   viennacl::slice eq(0, 1, ctx.num_of_eq); */
  /*   viennacl::slice start(0, ctx.nx, ctx.nz); */
  /*   viennacl::project(xn_im1, start, eq) = viennacl::project(xn, start, eq); */
  /* } */
  /* if (ctx.bc_left == comfi::types::PERIODIC) { */
  /*   viennacl::slice eq(0, 1, ctx.num_of_eq); */
  /*   viennacl::slice edge(inds(ctx.nx-1, 0, ctx), ctx.nx, ctx.nz); */
  /*   viennacl::slice start(0, ctx.nx, ctx.nz); */
  /*   viennacl::project(xn_im1, start, eq) = viennacl::project(xn, edge, eq); */
  /* } */

  static const sp_mat cpu_Pim1 = comfi::operators::buildPim1(ctx);
  static vcl_sp_mat Pim1(ctx.num_of_grid(), ctx.num_of_grid());
  static bool created = false;
  if (!created) { viennacl::copy(cpu_Pim1, Pim1); created = true; }
  vcl_mat xn_im1 = viennacl::linalg::prod(Pim1, xn);

  return xn_im1;
}

const sp_mat comfi::operators::buildPip1(comfi::types::Context &ctx) {
  umat locations;
  urowvec loci = zeros<urowvec>(ctx.num_of_grid());
  urowvec locj = zeros<urowvec>(ctx.num_of_grid());
  vec values = ones<vec>(ctx.num_of_grid());
  const comfi::types::BoundaryCondition BC = ctx.bc_right;

  #pragma omp parallel for collapse(2)
  for (uint i=0; i<ctx.nx; i++){ for(uint j=0; j<ctx.nz; j++) {
    // indexing
    const int           ij = inds(i, j, ctx);
    int                 ip1j = inds(i+1, j, ctx);

    //BC
    if (i==ctx.nx-1 && (BC==comfi::types::MIRROR || BC==comfi::types::NEUMANN)) { ip1j = ij; }
    else if (i==ctx.nx-1 && BC==comfi::types::PERIODIC) { ip1j = inds(0, j, ctx); }
    else if (BC==comfi::types::DIMENSIONLESS) { ip1j = inds(0, j, ctx); }

    loci(ij) = ip1j;
    locj(ij) = ij;
  }}

  locations.insert_rows(0, loci);
  locations.insert_rows(1, locj);
  return sp_mat(true, locations, values, ctx.num_of_grid(), ctx.num_of_grid());
}

const sp_mat comfi::operators::buildPim1(comfi::types::Context &ctx) {
  umat locations;
  urowvec loci = zeros<urowvec>(ctx.num_of_grid());
  urowvec locj = zeros<urowvec>(ctx.num_of_grid());
  vec values = ones<vec>(ctx.num_of_grid());
  const comfi::types::BoundaryCondition BC = ctx.bc_left;

  #pragma omp parallel for collapse(2)
  for (uint i=0; i<ctx.nx; i++){ for(uint j=0; j<ctx.nz; j++) {
    // indexing
    const int           ij = inds(i, j, ctx);
    int                 im1j = inds(i-1, j, ctx);

    //BC
    if (i==0 && (BC==comfi::types::MIRROR || BC==comfi::types::NEUMANN)) { im1j = ij; }
    else if (i==0 && BC==comfi::types::PERIODIC) { im1j = inds(ctx.nx-1, j, ctx);}
    else if (BC==comfi::types::DIMENSIONLESS) { im1j = inds(0, j, ctx);}

    loci(ij) = im1j;
    locj(ij) = ij;
  }}

  locations.insert_rows(0, loci);
  locations.insert_rows(1, locj);
  return sp_mat(true, locations, values, ctx.num_of_grid(), ctx.num_of_grid());
}

/*
vim: tabstop=2
vim: shiftwidth=2
vim: smarttab
vim: expandtab
*/
