#include "comfi.h"
#include <armadillo>
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/forwards.h"
using namespace arma;

vcl_mat comfi::operators::jm1(const vcl_mat &xn, comfi::types::Context ctx) {
  //make sure the dimension exists
  if (ctx.bc_down() == comfi::types::DIMENSIONLESS) { return xn; }
  //shift values
  vcl_mat xn_jm1 = viennacl::zero_matrix<double>(xn.size1(), xn.size2());
  viennacl::range eq(0, xn.size2());
  viennacl::range jm1(0, xn.size1()-ctx.nx());
  viennacl::range j(ctx.nx(), xn.size1());
  viennacl::project(xn_jm1, j, eq) = viennacl::project(xn, jm1, eq);
  //boundary conditions
  if (ctx.bc_down() == comfi::types::NEUMANN) {
    viennacl::range eq(0, xn.size2());
    viennacl::range edge(0, ctx.nx());
    viennacl::project(xn_jm1, edge, eq) = viennacl::project(xn, edge, eq);
  }
  if (ctx.bc_down() == comfi::types::PERIODIC) {
    viennacl::range eq(0, xn.size2());
    viennacl::range start(xn.size1()-ctx.nx(), xn.size1());
    viennacl::range edge(0, ctx.nx());
    viennacl::project(xn_jm1, edge, eq) = viennacl::project(xn, start, eq);
  }
  return xn_jm1;
}

vcl_mat comfi::operators::jp1(const vcl_mat &xn, comfi::types::Context ctx) {
  //make sure the dimension exists
  if (ctx.bc_up() == comfi::types::DIMENSIONLESS) { return xn; }
  //shift values
  vcl_mat xn_jp1 = viennacl::zero_matrix<double>(xn.size1(), xn.size2());
  viennacl::range eq(0, xn.size2());
  viennacl::range jp1(inds(0, 1, ctx), xn.size1());
  viennacl::range j(0, xn.size1()-ctx.nx());
  viennacl::project(xn_jp1, j, eq) = viennacl::project(xn, jp1, eq);
  //boundary conditions
  if (ctx.bc_up() == comfi::types::NEUMANN) {
    viennacl::range eq(0, xn.size2());
    viennacl::range edge(xn.size1()-ctx.nx(), xn.size1());
    viennacl::project(xn_jp1, edge, eq) = viennacl::project(xn, edge, eq);
  }
  if (ctx.bc_up() == comfi::types::PERIODIC) {
    viennacl::range eq(0, xn.size2());
    viennacl::range edge(xn.size1()-ctx.nx(), xn.size1());
    viennacl::range start(0, ctx.nx());
    viennacl::project(xn_jp1, edge, eq) = viennacl::project(xn, start, eq);
  }
  return xn_jp1;
}

vcl_mat comfi::operators::ip1(const vcl_mat &xn, comfi::types::Context ctx) {
  //make sure the dimension exists
  if (ctx.bc_right() == comfi::types::DIMENSIONLESS) { return xn; }
  //shift values
  vcl_mat xn_ip1 = viennacl::zero_matrix<double>(xn.size1(), xn.size2());
  viennacl::range eq(0, xn.size2());
  #pragma omp parallel for
  for(uint j = 0; j < ctx.nz(); j++) {
    viennacl::range ip1(inds(1, j, ctx), inds(ctx.nx()-1, j, ctx)+1);
    viennacl::range i(inds(0, j, ctx), inds(ctx.nx()-2, j, ctx)+1);
    viennacl::project(xn_ip1, i, eq) = viennacl::project(xn, ip1, eq);
  }
  //boundary conditions
  if (ctx.bc_right() == comfi::types::NEUMANN) {
    viennacl::slice eq(0, 1, xn.size2());
    viennacl::slice edge(inds(ctx.nx()-1, 0, ctx), ctx.nx(), ctx.nz());
    viennacl::project(xn_ip1, edge, eq) = viennacl::project(xn, edge, eq);
  }
  if (ctx.bc_right() == comfi::types::PERIODIC) {
    viennacl::slice eq(0, 1, xn.size2());
    viennacl::slice edge(inds(ctx.nx()-1, 0, ctx), ctx.nx(), ctx.nz());
    viennacl::slice start(0, ctx.nx(), ctx.nz());
    viennacl::project(xn_ip1, edge, eq) = viennacl::project(xn, start, eq);
  }
  return xn_ip1;
}

vcl_mat comfi::operators::im1(const vcl_mat &xn, comfi::types::Context ctx) {
  //make sure the dimension exists
  if (ctx.bc_left() == comfi::types::DIMENSIONLESS) { return xn; }
  //shift values
  vcl_mat xn_im1 = viennacl::zero_matrix<double>(xn.size1(), xn.size2());
  viennacl::range eq(0, xn.size2());
  #pragma omp parallel for
  for(uint j = 0; j < ctx.nz(); j++) {
    viennacl::range im1(inds(0, j, ctx), inds(ctx.nx()-2, j, ctx)+1);
    viennacl::range i(inds(1, j, ctx), inds(ctx.nx()-1, j, ctx)+1);
    viennacl::project(xn_im1, i, eq) = viennacl::project(xn, im1, eq);
  }
  //boundary conditions
  if (ctx.bc_left() == comfi::types::NEUMANN) {
    viennacl::slice eq(0, 1, xn.size2());
    viennacl::slice start(0, ctx.nx(), ctx.nz());
    viennacl::project(xn_im1, start, eq) = viennacl::project(xn, start, eq);
  }
  if (ctx.bc_left() == comfi::types::PERIODIC) {
    viennacl::slice eq(0, 1, xn.size2());
    viennacl::slice edge(inds(ctx.nx()-1, 0, ctx), ctx.nx(), ctx.nz());
    viennacl::slice start(0, ctx.nx(), ctx.nz());
    viennacl::project(xn_im1, start, eq) = viennacl::project(xn, edge, eq);
  }
  return xn_im1;
}

const sp_mat comfi::operators::buildPjp1(comfi::types::BoundaryCondition BC)
{
  const uint nnzp = num_of_eq;
  umat  Avi = zeros<umat>(nnzp, num_of_grid);
  umat  Avj = zeros<umat>(nnzp, num_of_grid);
  mat   Avv = zeros<mat> (nnzp, num_of_grid);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i
    const unsigned int  j=(index)/nx; // find j
    const int           ij=ind(i,j);

    int                 ijp1=ind(i,j+1);

    const double proportion=1.0;
    //BC
    if (j==nz-1 && BC==comfi::types::NEUMANN)
    {
        ijp1=ij;
        //proportion=0.1.0;
    }

    // Build sparse bulk vectors
    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=n_p+ij; vj(p)=n_p+ijp1;
    vv(p) = proportion;
    p++; vi(p)=n_n+ij; vj(p)=n_n+ijp1;
    vv(p) = proportion;
    p++; vi(p)=Vx+ij; vj(p)=Vx+ijp1;
    vv(p) = proportion;
    p++; vi(p)=Vz+ij; vj(p)=Vz+ijp1;
    vv(p) = proportion;
    p++; vi(p)=Vp+ij; vj(p)=Vp+ijp1;
    vv(p) = proportion;
    p++; vi(p)=Ux+ij; vj(p)=Ux+ijp1;
    vv(p) = proportion;
    p++; vi(p)=Uz+ij; vj(p)=Uz+ijp1;
    vv(p) = proportion;
    p++; vi(p)=Up+ij; vj(p)=Up+ijp1;
    vv(p) = proportion;
    p++; vi(p)=E_p+ij; vj(p)=E_p+ijp1;
    vv(p) = proportion;
    p++; vi(p)=E_n+ij; vj(p)=E_n+ijp1;
    vv(p) = proportion;
    p++; vi(p)=Bx+ij; vj(p)=Bx+ijp1;
    vv(p) = proportion;
    p++; vi(p)=Bz+ij; vj(p)=Bz+ijp1;
    vv(p) = proportion;
    p++; vi(p)=Bp+ij; vj(p)=Bp+ijp1;
    vv(p) = proportion;
    p++; vi(p)=GLM+ij; vj(p)=GLM+ijp1;
    vv(p) = proportion;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }
  // Reorder to create sparse matrix
  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::buildPjm1(comfi::types::BoundaryCondition BC)
{
  const uint nnzp= num_of_eq;
  umat  Avi = zeros<umat>(nnzp,num_of_grid);
  umat  Avj = zeros<umat>(nnzp,num_of_grid);
  mat   Avv = zeros<mat> (nnzp,num_of_grid);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i
    const unsigned int  j=(index)/nx; //find j
    const int           ij=ind(i,j);

    int                 ijm1=ind(i,j-1);

    //BC
    if (j==0 && BC==comfi::types::NEUMANN) { ijm1=ij; }
    else if (j == 0 && BC == comfi::types::PERIODIC) { ijm1 = ind(i, nz-1); }

    // Build sparse bulk vectors
    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=n_p+ij; vj(p)=n_p+ijm1;
    vv(p) = 1.0;
    p++; vi(p)=n_n+ij; vj(p)=n_n+ijm1;
    vv(p) = 1.0;
    p++; vi(p)=Vz+ij; vj(p)=Vz+ijm1;
    vv(p) = 1.0;
    //p++; vi(p)=Vp+ij; vj(p)=Vp+ijm1;
    //vv(p) = 1.0;
    p++; vi(p)=Ux+ij; vj(p)=Ux+ijm1;
    vv(p) = 1.0;
    p++; vi(p)=Uz+ij; vj(p)=Uz+ijm1;
    vv(p) = 1.0;
    //p++; vi(p)=Up+ij; vj(p)=Up+ijm1;
    //vv(p) = 1.0;
    p++; vi(p)=E_p+ij; vj(p)=E_p+ijm1;
    vv(p) = 1.0;
    p++; vi(p)=E_n+ij; vj(p)=E_n+ijm1;
    vv(p) = 1.0;
    p++; vi(p)=Bz+ij; vj(p)=Bz+ijm1;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=Bp+ijm1;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=Vp+ijm1;
    vv(p) = 1.0;
    p++; vi(p)=Up+ij; vj(p)=Up+ijm1;
    vv(p) = 1.0;
    p++; vi(p)=GLM+ij; vj(p)=GLM+ijm1;
    vv(p) = 1.0;
    p++; vi(p)=Bx+ij; vj(p)=Bx+ijm1;
    vv(p) = 1.0;
    //if (j!=0) {
    p++; vi(p)=Vx+ij; vj(p)=Vx+ijm1;
    vv(p) = 1.0;
    //}
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }
  // Reorder to create sparse matrix

  return comfi::util::syncSpMat(Avi, Avj, Avv); //reorder due to parallel construction
}

const sp_mat comfi::operators::buildPip1(comfi::types::BoundaryCondition BC)
{
  const uint nnzp = num_of_eq;
  umat  Avi = zeros<umat>(nnzp, num_of_grid);
  umat  Avj = zeros<umat>(nnzp, num_of_grid);
  mat   Avv = zeros<mat>(nnzp, num_of_grid);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i
    const unsigned int  j=(index)/nx; //find j
    const int           ij=ind(i,j);

    int                 ip1j=ind(i+1,j);

    //BC
    if (i==nx-1 && (BC==comfi::types::MIRROR || BC==comfi::types::NEUMANN)) { ip1j=ij; }
    if (i==nx-1 && BC==comfi::types::PERIODIC) { ip1j=ind(0,j); }
    if (BC==comfi::types::DIMENSIONLESS) { ip1j=ind(0, j); }

    // Build sparse bulk vectors
    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec>(nnzp);
    p++; vi(p)=n_p+ij; vj(p)=n_p+ip1j;
    vv(p) = 1.0;
    p++; vi(p)=n_n+ij; vj(p)=n_n+ip1j;
    vv(p) = 1.0;
    p++; vi(p)=Vz+ij; vj(p)=Vz+ip1j;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=Vp+ip1j;
    vv(p) = 1.0;
    if (i==nx-1 && BC==comfi::types::MIRROR)
    {
      p++; vi(p)=Bx+ij; vj(p)=Bx+ip1j;
      vv(p) = -1.0;
      p++; vi(p)=Vx+ij; vj(p)=Vx+ip1j;
      vv(p) = -1.0;
      p++; vi(p)=Ux+ij; vj(p)=Ux+ip1j;
      vv(p) = -1.0;
    } else {
      p++; vi(p)=Bx+ij; vj(p)=Bx+ip1j;
      vv(p) = 1.0;
      p++; vi(p)=Vx+ij; vj(p)=Vx+ip1j;
      vv(p) = 1.0;
      p++; vi(p)=Ux+ij; vj(p)=Ux+ip1j;
      vv(p) = 1.0;
    }
    p++; vi(p)=Uz+ij; vj(p)=Uz+ip1j;
    vv(p) = 1.0;
    p++; vi(p)=Up+ij; vj(p)=Up+ip1j;
    vv(p) = 1.0;
    p++; vi(p)=E_p+ij; vj(p)=E_p+ip1j;
    vv(p) = 1.0;
    p++; vi(p)=E_n+ij; vj(p)=E_n+ip1j;
    vv(p) = 1.0;
    p++; vi(p)=Bz+ij; vj(p)=Bz+ip1j;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=Bp+ip1j;
    vv(p) = 1.0;
    p++; vi(p)=GLM+ij; vj(p)=GLM+ip1j;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }
  // Reorder to create sparse matrix
  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::buildPim1(comfi::types::BoundaryCondition BC)
{
  const uint nnzp= num_of_eq;
  umat  Avi = zeros<umat>(nnzp,num_of_grid);
  umat  Avj = zeros<umat>(nnzp,num_of_grid);
  mat   Avv = zeros<mat> (nnzp,num_of_grid);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i
    const unsigned int  j=(index)/nx; //find j
    const int           ij=ind(i,j);

    int                 im1j=ind(i-1,j);

    //BC
    if (i==0 && (BC==comfi::types::MIRROR || BC==comfi::types::NEUMANN)) { im1j = ij; }
    else if (i==0 && BC==comfi::types::PERIODIC) { im1j = ind(nx-1, j);}
    else if (BC==comfi::types::DIMENSIONLESS) { im1j = ind(0, j);}

    // Build sparse bulk vectors
    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=n_p+ij; vj(p)=n_p+im1j;
    vv(p) = 1.0;
    p++; vi(p)=n_n+ij; vj(p)=n_n+im1j;
    vv(p) = 1.0;
    p++; vi(p)=Vz+ij; vj(p)=Vz+im1j;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=Vp+im1j;
    vv(p) = 1.0;
    p++; vi(p)=Uz+ij; vj(p)=Uz+im1j;
    vv(p) = 1.0;
    p++; vi(p)=Up+ij; vj(p)=Up+im1j;
    vv(p) = 1.0;
    p++; vi(p)=E_p+ij; vj(p)=E_p+im1j;
    vv(p) = 1.0;
    p++; vi(p)=E_n+ij; vj(p)=E_n+im1j;
    vv(p) = 1.0;
    p++; vi(p)=Bz+ij; vj(p)=Bz+im1j;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=Bp+im1j;
    vv(p) = 1.0;
    p++; vi(p)=GLM+ij; vj(p)=GLM+im1j;
    vv(p) = 1.0;
    if (i==0 && BC==comfi::types::MIRROR)
    {
      p++; vi(p)=Vx+ij; vj(p)=Vx+im1j;
      vv(p) = -1.0;
      p++; vi(p)=Ux+ij; vj(p)=Ux+im1j;
      vv(p) = -1.0;
      p++; vi(p)=Bx+ij; vj(p)=Bx+im1j;
      vv(p) = -1.0;
    } else {
      p++; vi(p)=Bx+ij; vj(p)=Bx+im1j;
      vv(p) = 1.0;
      p++; vi(p)=Vx+ij; vj(p)=Vx+im1j;
      vv(p) = 1.0;
      p++; vi(p)=Ux+ij; vj(p)=Ux+im1j;
      vv(p) = 1.0;
    }
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }
  // Reorder to create sparse matrix
  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::buildPip2(comfi::types::BoundaryCondition BC)
{
  const uint nnzp= num_of_eq;
  umat  Avi = zeros<umat>(nnzp,num_of_grid);
  umat  Avj = zeros<umat>(nnzp,num_of_grid);
  mat   Avv = zeros<mat> (nnzp,num_of_grid);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i
    const unsigned int  j=(index)/nx; //find j
    const int           ij=ind(i,j);

    const int           ip1j=ind(i+1,j);
    const int           im1j=ind(i-1,j);
    int                 ip2j=ind(i+2,j);

    //BC
    if (i==(nx-1) && BC==comfi::types::MIRROR) { ip2j = im1j; }
    else if (i==(nx-1) && BC==comfi::types::NEUMANN) { ip2j = ij; }
    else if (i==(nx-1) && BC==comfi::types::PERIODIC) { ip2j = ind(1, j); }
    else if (BC==comfi::types::DIMENSIONLESS) { ip2j = ind(0, j);}
    else if (i==(nx-2) && (BC==comfi::types::MIRROR || BC==comfi::types::NEUMANN)) { ip2j=ip1j; }
    else if (i==(nx-2) && BC==comfi::types::PERIODIC) { ip2j = ind(0, j); }

    // Build sparse bulk vectors
    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=n_p+ij; vj(p)=n_p+ip2j;
    vv(p) = 1.0;
    p++; vi(p)=n_n+ij; vj(p)=n_n+ip2j;
    vv(p) = 1.0;
    p++; vi(p)=Vz+ij; vj(p)=Vz+ip2j;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=Vp+ip2j;
    vv(p) = 1.0;
    p++; vi(p)=Uz+ij; vj(p)=Uz+ip2j;
    vv(p) = 1.0;
    p++; vi(p)=Up+ij; vj(p)=Up+ip2j;
    vv(p) = 1.0;
    p++; vi(p)=E_p+ij; vj(p)=E_p+ip2j;
    vv(p) = 1.0;
    p++; vi(p)=E_n+ij; vj(p)=E_n+ip2j;
    vv(p) = 1.0;
    p++; vi(p)=Bz+ij; vj(p)=Bz+ip2j;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=Bp+ip2j;
    vv(p) = 1.0;
    p++; vi(p)=GLM+ij; vj(p)=GLM+ip2j;
    vv(p) = 1.0;
    if ((i==(nx-1) || i==(nx-2)) && BC==comfi::types::MIRROR)
    {
      p++; vi(p)=Vx+ij; vj(p)=Vx+ip2j;
      vv(p) = -1.0;
      p++; vi(p)=Ux+ij; vj(p)=Ux+ip2j;
      vv(p) = -1.0;
      p++; vi(p)=Bx+ij; vj(p)=Bx+ip2j;
      vv(p) = -1.0;
    } else {
      p++; vi(p)=Ux+ij; vj(p)=Ux+ip2j;
      vv(p) = 1.0;
      p++; vi(p)=Vx+ij; vj(p)=Vx+ip2j;
      vv(p) = 1.0;
      p++; vi(p)=Bx+ij; vj(p)=Bx+ip2j;
      vv(p) = 1.0;
    }

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::buildPim2(comfi::types::BoundaryCondition BC)
{
  const uint nnzp= num_of_eq;
  umat  Avi = zeros<umat>(nnzp,num_of_grid);
  umat  Avj = zeros<umat>(nnzp,num_of_grid);
  mat   Avv = zeros<mat> (nnzp,num_of_grid);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i
    const unsigned int  j=(index)/nx; //find j
    const int           ij=ind(i, j);

    const int           im1j=ind(i-1, j);
    const int           ip1j=ind(i+1, j);
    int                 im2j=ind(i-2, j);

    //BC
    if (i==1 && (BC==comfi::types::MIRROR || BC==comfi::types::NEUMANN)) { im2j=im1j; }
    else if (i==1 && BC==comfi::types::PERIODIC) { im2j = ind(nx-1, j); }
    else if (i==0 && BC==comfi::types::MIRROR) { im2j = ip1j; }
    else if (i==0 && BC==comfi::types::NEUMANN) { im2j = ij; }
    else if (BC==comfi::types::DIMENSIONLESS) { im2j = ind(0, j); }
    else if (i==0 && BC==comfi::types::PERIODIC) { im2j = ind(nx-2, j); }

    // Build sparse bulk vectors
    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=n_p+ij; vj(p)=n_p+im2j;
    vv(p) = 1.0;
    p++; vi(p)=n_n+ij; vj(p)=n_n+im2j;
    vv(p) = 1.0;
    p++; vi(p)=Vz+ij; vj(p)=Vz+im2j;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=Vp+im2j;
    vv(p) = 1.0;
    p++; vi(p)=Uz+ij; vj(p)=Uz+im2j;
    vv(p) = 1.0;
    p++; vi(p)=Up+ij; vj(p)=Up+im2j;
    vv(p) = 1.0;
    p++; vi(p)=E_p+ij; vj(p)=E_p+im2j;
    vv(p) = 1.0;
    p++; vi(p)=E_n+ij; vj(p)=E_n+im2j;
    vv(p) = 1.0;
    p++; vi(p)=Bz+ij; vj(p)=Bz+im2j;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=Bp+im2j;
    vv(p) = 1.0;
    p++; vi(p)=GLM+ij; vj(p)=GLM+im2j;
    vv(p) = 1.0;
    if ((i==0 || i==1) && BC==comfi::types::MIRROR)
    {
      p++; vi(p)=Ux+ij; vj(p)=Ux+im2j;
      vv(p) = -1.0;
      p++; vi(p)=Vx+ij; vj(p)=Vx+im2j;
      vv(p) = -1.0;
      p++; vi(p)=Bx+ij; vj(p)=Bx+im2j;
      vv(p) = -1.0;
    } else {
      p++; vi(p)=Ux+ij; vj(p)=Ux+im2j;
      vv(p) = 1.0;
      p++; vi(p)=Vx+ij; vj(p)=Vx+im2j;
      vv(p) = 1.0;
      p++; vi(p)=Bx+ij; vj(p)=Bx+im2j;
      vv(p) = 1.0;
    }
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::buildPjp2(comfi::types::BoundaryCondition BC)
{
  const uint nnzp= num_of_eq;
  umat  Avi = zeros<umat>(nnzp,num_of_grid);
  umat  Avj = zeros<umat>(nnzp,num_of_grid);
  mat   Avv = zeros<mat> (nnzp,num_of_grid);

  // asynch parallel sparse matrix mhdsim::operators::building
  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i
    const unsigned int  j=(index)/nx; //find j
    const int           ij=ind(i,j);

    int                 ijp2=ind(i,j+2);
    int                 ijp1=ind(i,j+1);

    double proportion = 1.0;
    //BC
    if (j==(nz-1) && BC==comfi::types::NEUMANN){
      ijp2=ij;
      //proportion = 0.01.0;
    } else if (j==(nz-2) && BC==comfi::types::NEUMANN) {
      ijp2=ijp1;
      //proportion = 0.1.0;
    }

    // Build sparse bulk vectors
    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=n_p+ij; vj(p)=n_p+ijp2;
    vv(p) = proportion;
    p++; vi(p)=n_n+ij; vj(p)=n_n+ijp2;
    vv(p) = proportion;
    p++; vi(p)=Vx+ij; vj(p)=Vx+ijp2;
    vv(p) = proportion;
    p++; vi(p)=Vz+ij; vj(p)=Vz+ijp2;
    vv(p) = proportion;
    p++; vi(p)=Vp+ij; vj(p)=Vp+ijp2;
    vv(p) = proportion;
    p++; vi(p)=Ux+ij; vj(p)=Ux+ijp2;
    vv(p) = proportion;
    p++; vi(p)=Uz+ij; vj(p)=Uz+ijp2;
    vv(p) = proportion;
    p++; vi(p)=Up+ij; vj(p)=Up+ijp2;
    vv(p) = proportion;
    p++; vi(p)=E_p+ij; vj(p)=E_p+ijp2;
    vv(p) = proportion;
    p++; vi(p)=E_n+ij; vj(p)=E_n+ijp2;
    vv(p) = proportion;
    p++; vi(p)=Bx+ij; vj(p)=Bx+ijp2;
    vv(p) = proportion;
    p++; vi(p)=Bz+ij; vj(p)=Bz+ijp2;
    vv(p) = proportion;
    p++; vi(p)=Bp+ij; vj(p)=Bp+ijp2;
    vv(p) = proportion;
    p++; vi(p)=GLM+ij; vj(p)=GLM+ijp2;
    vv(p) = proportion;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::buildPjm2(comfi::types::BoundaryCondition BC)
{
  const uint nnzp= num_of_eq;
  umat  Avi = zeros<umat>(nnzp,num_of_grid);
  umat  Avj = zeros<umat>(nnzp,num_of_grid);
  mat   Avv = zeros<mat> (nnzp,num_of_grid);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i
    const unsigned int  j=(index)/nx; //find j
    const int           ij=ind(i,j);

    int                 ijm2=ind(i,j-2);
    int                 ijm1=ind(i,j-1);

    //BC
    if (j==0 && BC==comfi::types::NEUMANN) { ijm2=ij; }
    else if (j==1 && BC==comfi::types::NEUMANN) { ijm2=ijm1; }

    // Build sparse bulk vectors
    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=n_p+ij; vj(p)=n_p+ijm2;
    vv(p) = 1.0;
    p++; vi(p)=n_n+ij; vj(p)=n_n+ijm2;
    vv(p) = 1.0;
    p++; vi(p)=Vz+ij; vj(p)=Vz+ijm2;
    vv(p) = 1.0;
    //p++; vi(p)=Vp+ij; vj(p)=Vp+ijm2;
    //vv(p) = 1.0;
    p++; vi(p)=Ux+ij; vj(p)=Ux+ijm2;
    vv(p) = 1.0;
    p++; vi(p)=Uz+ij; vj(p)=Uz+ijm2;
    vv(p) = 1.0;
    //p++; vi(p)=Up+ij; vj(p)=Up+ijm2;
    //vv(p) = 1.0;
    p++; vi(p)=E_p+ij; vj(p)=E_p+ijm2;
    vv(p) = 1.0;
    p++; vi(p)=E_n+ij; vj(p)=E_n+ijm2;
    vv(p) = 1.0;
    p++; vi(p)=Bz+ij; vj(p)=Bz+ijm2;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=Bp+ijm2;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=Vp+ijm2;
    vv(p) = 1.0;
    p++; vi(p)=Up+ij; vj(p)=Up+ijm2;
    vv(p) = 1.0;
    p++; vi(p)=GLM+ij; vj(p)=GLM+ijm2;
    vv(p) = 1.0;
    p++; vi(p)=Bx+ij; vj(p)=Bx+ijm2;
    vv(p) = 1.0;
    //if (j>1){
    p++; vi(p)=Vx+ij; vj(p)=Vx+ijm2;
    vv(p) = 1.0;
    //}
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }
  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::buildSG()
{
  const uint nnzp= 2;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=Vz+ij; vj(p)=n_p+ij;
    vv(p) = g;
    p++; vi(p)=Uz+ij; vj(p)=n_n+ij;
    vv(p) = g;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::buildPEigVx()
{
  const uint nnzp= 13;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=n_p+ij; vj(p)=Vx+ij;
    vv(p) = 1.0;
    p++; vi(p)=n_n+ij; vj(p)=Ux+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vx+ij; vj(p)=Vx+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vz+ij; vj(p)=Vx+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=Vx+ij;
    vv(p) = 1.0;
    p++; vi(p)=Ux+ij; vj(p)=Ux+ij;
    vv(p) = 1.0;
    p++; vi(p)=Uz+ij; vj(p)=Ux+ij;
    vv(p) = 1.0;
    p++; vi(p)=Up+ij; vj(p)=Ux+ij;
    vv(p) = 1.0;
    p++; vi(p)=E_p+ij; vj(p)=Vx+ij;
    vv(p) = 1.0;
    p++; vi(p)=E_n+ij; vj(p)=Ux+ij;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=Vx+ij;
    vv(p) = 1.0;
    p++; vi(p)=Bz+ij; vj(p)=Vx+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::buildPEigVz()
{
  const uint nnzp = 13;
  umat  Avi = zeros<umat>(nnzp, num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp, num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp, num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=n_p+ij; vj(p)=Vz+ij;
    vv(p) = 1.0;
    p++; vi(p)=n_n+ij; vj(p)=Uz+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vx+ij; vj(p)=Vz+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vz+ij; vj(p)=Vz+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=Vz+ij;
    vv(p) = 1.0;
    p++; vi(p)=Ux+ij; vj(p)=Uz+ij;
    vv(p) = 1.0;
    p++; vi(p)=Uz+ij; vj(p)=Uz+ij;
    vv(p) = 1.0;
    p++; vi(p)=Up+ij; vj(p)=Uz+ij;
    vv(p) = 1.0;
    p++; vi(p)=E_p+ij; vj(p)=Vz+ij;
    vv(p) = 1.0;
    p++; vi(p)=E_n+ij; vj(p)=Uz+ij;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=Vz+ij;
    vv(p) = 1.0;
    p++; vi(p)=Bx+ij; vj(p)=Vz+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::buildPN()
{
  const uint nnzp= num_of_eq;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=n_p+ij; vj(p)=n_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=n_n+ij; vj(p)=n_n+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vx+ij; vj(p)=n_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vz+ij; vj(p)=n_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=n_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=Ux+ij; vj(p)=n_n+ij;
    vv(p) = 1.0;
    p++; vi(p)=Uz+ij; vj(p)=n_n+ij;
    vv(p) = 1.0;
    p++; vi(p)=Up+ij; vj(p)=n_n+ij;
    vv(p) = 1.0;
    p++; vi(p)=E_p+ij; vj(p)=n_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=E_n+ij; vj(p)=n_n+ij;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=n_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=Bx+ij; vj(p)=n_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=Bz+ij; vj(p)=n_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=GLM+ij; vj(p)=n_p+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::builds2Tp()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=E_p+ij; vj(p)=sij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,num_of_grid);
}

const sp_mat comfi::operators::builds2Vx()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=Vx+ij; vj(p)=sij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,num_of_grid);
}

const sp_mat comfi::operators::builds2Vz()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=Vz+ij; vj(p)=sij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,num_of_grid);
}

const sp_mat comfi::operators::builds2Ux()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=Ux+ij; vj(p)=sij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,num_of_grid);
}

const sp_mat comfi::operators::builds2Uz()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=Uz+ij; vj(p)=sij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,num_of_grid);
}

const sp_mat comfi::operators::builds2Tn()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=E_n+ij; vj(p)=sij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,num_of_grid);
}

const sp_mat comfi::operators::builds2Np()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=n_p+ij; vj(p)=sij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,num_of_grid);
}

const sp_mat comfi::operators::builds2Nn()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=n_n+ij; vj(p)=sij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,num_of_grid);
}

const sp_mat comfi::operators::builds2Bx()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=Bx+ij; vj(p)=sij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,num_of_grid);
}

const sp_mat comfi::operators::builds2Bz()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=Bz+ij; vj(p)=sij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,num_of_grid);
}

const sp_mat comfi::operators::builds2GLM()
{
  const uint nnzp = 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=GLM+ij; vj(p)=sij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,num_of_grid);
}

const sp_mat comfi::operators::buildCurl(comfi::types::BoundaryCondition LeftBC,
                                          comfi::types::BoundaryCondition RightBC,
                                          comfi::types::BoundaryCondition UpBC,
                                          comfi::types::BoundaryCondition DownBC)
{
  const uint nnzp = 13;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           fij=indf(i,j);
    int                 fip1j=indf(i+1,j);
    int                 fim1j=indf(i-1,j);
    int                 fijp1=indf(i,j+1);
    int                 fijm1=indf(i,j-1);

    if (j==(nz-1) && UpBC==comfi::types::NEUMANN)
    {
      fijp1=fij;
    } else if (j==0 && DownBC==comfi::types::NEUMANN) {
      fijm1=fij;
    }
    if (i==(nx-1) && (RightBC==comfi::types::NEUMANN || RightBC==comfi::types::MIRROR))
    {
      fip1j=fij;
    } else if (i==0 && (LeftBC==comfi::types::NEUMANN || LeftBC==comfi::types::MIRROR)) {
      fim1j=fij;
    } else if (i==(nx-1) && (RightBC==comfi::types::PERIODIC)) {
      fip1j=indf(0, j);
    } else if (i==0 && (LeftBC==comfi::types::PERIODIC)) {
      fim1j=indf(nx-1, j);
    }
    if (RightBC == comfi::types::DIMENSIONLESS){ fip1j = indf(0, j);}
    if (LeftBC == comfi::types::DIMENSIONLESS){ fim1j = indf(0, j);}


    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec vv = zeros<vec>(nnzp);

    p++; vi(p)=_x+fij; vj(p)=_p+fijm1;
    vv(p) = -0.5/dz;
    p++; vi(p)=_x+fij; vj(p)=_p+fijp1;
    vv(p) = 0.5/dz;
    p++; vi(p)=_p+fij; vj(p)=_x+fijp1;
    vv(p) = -0.5/dz;
    p++; vi(p)=_p+fij; vj(p)=_x+fijm1;
    vv(p) = 0.5/dz;
    p++; vi(p)=_p+fij; vj(p)=_z+fim1j;
    vv(p) = -0.5/dx;
    p++; vi(p)=_p+fij; vj(p)=_z+fip1j;
    vv(p) = 0.5/dx;
    p++; vi(p)=_z+fij; vj(p)=_p+fip1j;
    vv(p) = -0.5/dx;
    p++; vi(p)=_z+fij; vj(p)=_p+fim1j;
    vv(p) = 0.5/dx;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,DOF*num_of_grid,DOF*num_of_grid);
}

const sp_mat comfi::operators::buildCross1()
{
  const uint nnzp= 3;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           fij=indf(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=_x+fij; vj(p)=_z+fij;
    vv(p) = 1.0;
    p++; vi(p)=_p+fij; vj(p)=_x+fij;
    vv(p) = 1.0;
    p++; vi(p)=_z+fij; vj(p)=_p+fij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,DOF*num_of_grid,DOF*num_of_grid);
}

const sp_mat comfi::operators::buildCross2()
{
  const uint nnzp= 3;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           fij=indf(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=_x+fij; vj(p)=_p+fij;
    vv(p) = 1.0;
    p++; vi(p)=_p+fij; vj(p)=_z+fij;
    vv(p) = 1.0;
    p++; vi(p)=_z+fij; vj(p)=_x+fij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,DOF*num_of_grid,DOF*num_of_grid);
}

const sp_mat comfi::operators::buildNnscalar()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=sij; vj(p)=n_n+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_grid,num_of_elem);
}

const sp_mat comfi::operators::buildNpscalar()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=sij; vj(p)=n_p+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_grid,num_of_elem);
}

const sp_mat comfi::operators::buildTpscalar()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=sij; vj(p)=E_p+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_grid,num_of_elem);
}

const sp_mat comfi::operators::buildTnscalar()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=sij; vj(p)=E_n+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_grid,num_of_elem);
}

const sp_mat comfi::operators::buildGLMscalar()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=sij; vj(p)=GLM+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_grid,num_of_elem);
}

const sp_mat comfi::operators::buildpFBB()
{
  const uint nnzp= 3;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           fij=indf(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=Vx+ij; vj(p)=_x+fij;
    vv(p) = 1.0;
    p++; vi(p)=Vz+ij; vj(p)=_z+fij;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=_p+fij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,DOF*num_of_grid);
}

const sp_mat comfi::operators::buildBfield()
{
  const uint nnzp= 3;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);      // global index
    const int           fij=indf(i,j);    // mhdsim::operators::field index

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=_x+fij; vj(p)=Bx+ij;
    vv(p) = 1.0;
    p++; vi(p)=_z+fij; vj(p)=Bz+ij;
    vv(p) = 1.0;
    p++; vi(p)=_p+fij; vj(p)=Bp+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,DOF*num_of_grid,num_of_elem);
}

const sp_mat comfi::operators::buildVfield()
{
  const uint nnzp= 3;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);      // global index
    const int           fij=indf(i,j);    // mhdsim::operators::field index

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=_x+fij; vj(p)=Vx+ij;
    vv(p) = 1.0;
    p++; vi(p)=_z+fij; vj(p)=Vz+ij;
    vv(p) = 1.0;
    p++; vi(p)=_p+fij; vj(p)=Vp+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,DOF*num_of_grid,num_of_elem);
}

const sp_mat comfi::operators::buildUfield()
{
  const uint nnzp= 3;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);      // global index
    const int           fij=indf(i,j);    // mhdsim::operators::field index

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=_x+fij; vj(p)=Ux+ij;
    vv(p) = 1.0;
    p++; vi(p)=_z+fij; vj(p)=Uz+ij;
    vv(p) = 1.0;
    p++; vi(p)=_p+fij; vj(p)=Up+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,DOF*num_of_grid,num_of_elem);
}

const sp_mat comfi::operators::buildFxVB()
{
  const uint nnzp= 2;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           fij=indf(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=Bz+ij; vj(p)=_z+fij;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=_p+fij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,DOF*num_of_grid);
}

const sp_mat comfi::operators::buildf2B()
{
  const uint nnzp= 3;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           fij=indf(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=Bz+ij; vj(p)=_z+fij;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=_p+fij;
    vv(p) = 1.0;
    p++; vi(p)=Bx+ij; vj(p)=_x+fij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,DOF*num_of_grid);
}

const sp_mat comfi::operators::buildf2V()
{
  const uint nnzp= 3;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           fij=indf(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=Vz+ij; vj(p)=_z+fij;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=_p+fij;
    vv(p) = 1.0;
    p++; vi(p)=Vx+ij; vj(p)=_x+fij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,DOF*num_of_grid);
}

const sp_mat comfi::operators::buildf2U()
{
  const uint nnzp= 3;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           fij=indf(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=Uz+ij; vj(p)=_z+fij;
    vv(p) = 1.0;
    p++; vi(p)=Up+ij; vj(p)=_p+fij;
    vv(p) = 1.0;
    p++; vi(p)=Ux+ij; vj(p)=_x+fij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,DOF*num_of_grid);
}

const sp_mat comfi::operators::buildFzVB()
{
  const uint nnzp= 2;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    const int           fij=indf(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=Bx+ij; vj(p)=_x+fij;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=_p+fij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_elem,DOF*num_of_grid);
}

const sp_mat comfi::operators::buildTopBC()
{
  const uint nnzp = 8;
  umat  Avi = zeros<umat>(nnzp, nx); // store row index
  umat  Avj = zeros<umat>(nnzp, nx); // store col index
  mat   Avv = zeros<mat>(nnzp, nx); // store value

  #pragma omp parallel for schedule(static)
  for (uint i=0; i<nx; i++)
  {
    //indexing
    const int           ij = ind(i, nz-1);

    int p = -1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec>(nnzp);

    p++; vi(p)=n_p+ij; vj(p)=n_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=E_p+ij; vj(p)=E_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=Vp+ij;
    vv(p) = 1.0;
    p++; vi(p)=n_n+ij; vj(p)=n_n+ij;
    vv(p) = 1.0;
    p++; vi(p)=E_n+ij; vj(p)=E_n+ij;
    vv(p) = 1.0;
    p++; vi(p)=Up+ij; vj(p)=Up+ij;
    vv(p) = 1.0;
    p++; vi(p)=Uz+ij; vj(p)=Uz+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vz+ij; vj(p)=Vz+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(i) = vi;
    Avj.col(i) = vj;
    Avv.col(i) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::buildBottomBC()
{
  const uint nnzp = 8;
  umat  Avi = zeros<umat>(nnzp, nx); // store row index
  umat  Avj = zeros<umat>(nnzp, nx); // store col index
  mat   Avv = zeros<mat>(nnzp, nx); // store value

  #pragma omp parallel for schedule(static)
  for (uint i=0; i<nx; i++)
  {
    //indexing
    const int           ij = ind(i, 0);

    int p = -1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec>(nnzp);

    p++; vi(p)=n_p+ij; vj(p)=n_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=E_p+ij; vj(p)=E_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vp+ij; vj(p)=Vp+ij;
    vv(p) = 1.0;
    p++; vi(p)=n_n+ij; vj(p)=n_n+ij;
    vv(p) = 1.0;
    p++; vi(p)=E_n+ij; vj(p)=E_n+ij;
    vv(p) = 1.0;
    p++; vi(p)=Up+ij; vj(p)=Up+ij;
    vv(p) = 1.0;
    p++; vi(p)=Uz+ij; vj(p)=Uz+ij;
    vv(p) = 1.0;
    p++; vi(p)=Vz+ij; vj(p)=Vz+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(i) = vi;
    Avj.col(i) = vj;
    Avv.col(i) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::buildT()
{
  const uint nnzp= 2;
  umat  Avi = zeros<umat>(nnzp, num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp, num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp, num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const unsigned int  ij=ind(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=E_p+ij; vj(p)=E_p+ij;
    vv(p) = 1.0;
    p++; vi(p)=E_n+ij; vj(p)=E_n+ij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv);
}

const sp_mat comfi::operators::field_scalarGrad(comfi::types::BoundaryCondition leftBC,
                                                 comfi::types::BoundaryCondition bottomBC,
                                                 comfi::types::BoundaryCondition topBC,
                                                 comfi::types::BoundaryCondition rightBC)
{
  const uint nnzp = 4;
  umat  Avi = zeros<umat>(nnzp, num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp, num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp, num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           fij=indf(i,j);
    const int           sij=inds(i,j);
    int                 sip1j=inds(i+1,j);
    int                 sim1j=inds(i-1,j);
    int                 sijp1=inds(i,j+1);
    int                 sijm1=inds(i,j-1);

    if ((i == 0) && (leftBC == comfi::types::MIRROR || leftBC ==  comfi::types::NEUMANN))
      { sim1j = sij; }
    else if ((i == nx-1) && (rightBC == comfi::types::MIRROR || rightBC == comfi::types::NEUMANN))
      { sip1j = sij; }
    else if (i == 0 && leftBC == comfi::types::PERIODIC)
      { sim1j = inds(nx-1, j); }
    else if (i == (nx-1) && rightBC == comfi::types::PERIODIC)
      { sip1j = inds(0, j); }
    if ((j == 0) && (bottomBC == comfi::types::MIRROR || bottomBC == comfi::types::NEUMANN))
      { sijm1=sij; }
    else if ((j == nz-1) && (topBC == comfi::types::MIRROR || topBC == comfi::types::NEUMANN))
      { sijp1=sij; }
    if (leftBC == comfi::types::DIMENSIONLESS){ sim1j = inds(0, j);}
    if (rightBC == comfi::types::DIMENSIONLESS){ sip1j = inds(0, j);}

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    if (rightBC != comfi::types::DIRICHLET) {
      p++; vi(p)=_x+fij; vj(p)=sip1j;
      vv(p) = 0.5/dx;
    }
    if (leftBC != comfi::types::DIRICHLET) {
    p++; vi(p)=_x+fij; vj(p)=sim1j;
    vv(p) = -0.5/dx;
    }
    if (topBC != comfi::types::DIRICHLET) {
    p++; vi(p)=_z+fij; vj(p)=sijp1;
    vv(p) = 0.5/dz;
    }
    if (bottomBC != comfi::types::DIRICHLET) {
    p++; vi(p)=_z+fij; vj(p)=sijm1;
    vv(p) = -0.5/dz;
    }
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv, DOF*num_of_grid, num_of_grid);
}

const sp_mat comfi::operators::field_xProjection()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp, num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp, num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp, num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           fij=indf(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=sij; vj(p)=_x+fij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_grid,DOF*num_of_grid);
}

const sp_mat comfi::operators::field_zProjection()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           fij=indf(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=sij; vj(p)=_z+fij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_grid,DOF*num_of_grid);
}

const sp_mat comfi::operators::field_pProjection()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           fij=indf(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=sij; vj(p)=_p+fij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_grid,DOF*num_of_grid);
}

const sp_mat comfi::operators::field_scalar2xfield()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           fij=indf(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=_x+fij; vj(p)=sij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,DOF*num_of_grid,num_of_grid);
}
const sp_mat comfi::operators::field_scalar2pfield()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           fij=indf(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=_p+fij; vj(p)=sij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,DOF*num_of_grid,num_of_grid);
}
const sp_mat comfi::operators::field_scalar2zfield()
{
  const uint nnzp= 1.0;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           fij=indf(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=_z+fij; vj(p)=sij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,DOF*num_of_grid,num_of_grid);
}

const sp_mat comfi::operators::field_scalar2field()
{
  const uint nnzp= 3;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           fij=indf(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=_x+fij; vj(p)=sij;
    vv(p) = 1.0;
    p++; vi(p)=_z+fij; vj(p)=sij;
    vv(p) = 1.0;
    p++; vi(p)=_p+fij; vj(p)=sij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,DOF*num_of_grid,num_of_grid);
}

const sp_mat comfi::operators::field_field2scalar()
{
  const uint nnzp= 3;
  umat  Avi = zeros<umat>(nnzp,num_of_grid); // store row index
  umat  Avj = zeros<umat>(nnzp,num_of_grid); // store col index
  mat   Avv = zeros<mat> (nnzp,num_of_grid); // store value

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           fij=indf(i,j);
    const int           sij=inds(i,j);

    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++;  vi(p)=sij; vj(p)=_x+fij;
    vv(p) = 1.0;
    p++;  vi(p)=sij; vj(p)=_z+fij;
    vv(p) = 1.0;
    p++;  vi(p)=sij; vj(p)=_p+fij;
    vv(p) = 1.0;
    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_grid,DOF*num_of_grid);
}

const sp_mat comfi::operators::field_fieldDiv(comfi::types::BoundaryCondition LeftBC,
                                               comfi::types::BoundaryCondition RightBC,
                                               comfi::types::BoundaryCondition UpBC,
                                               comfi::types::BoundaryCondition DownBC)
{
  const uint nnzp= 5;
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
    const int           fij=indf(i,j);

    int                 fijm1=indf(i,j-1);
    int                 fijp1=indf(i,j+1);
    int                 fip1j=indf(i+1,j);
    int                 fim1j=indf(i-1,j);
    //BC
    if (j==0 && DownBC==comfi::types::NEUMANN) { fijm1=fij; } else
    if (j==(nz-1) && UpBC==comfi::types::NEUMANN) { fijp1=fij; }
    if (i==0 && (LeftBC==comfi::types::MIRROR || LeftBC==comfi::types::NEUMANN)) { fim1j=fij; } else
    if (i==0 && (LeftBC==comfi::types::PERIODIC)) { fim1j=indf(nx-1,j); } else
    if (i==(nx-1) && (RightBC==comfi::types::MIRROR || RightBC==comfi::types::NEUMANN)) { fip1j=fij; } else
    if (i==(nx-1) && RightBC==comfi::types::PERIODIC) { fip1j=indf(0,j); }
    if (LeftBC == comfi::types::DIMENSIONLESS) { fim1j = indf(0, j); }
    if (RightBC == comfi::types::DIMENSIONLESS) { fip1j = indf(0, j); }

    // Build sparse bulk vectors
    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);
    p++; vi(p)=sij; vj(p)=_x+fip1j;
    if (i==(nx-1) && RightBC==comfi::types::MIRROR) { vv(p) = -0.5/dx; } else { vv(p) = 0.5/dx; }
    p++; vi(p)=sij; vj(p)=_x+fim1j;
    if (i==0 && LeftBC==comfi::types::MIRROR) { vv(p) = 0.5/dx; } else { vv(p) = -0.5/dx; }
    p++; vi(p)=sij; vj(p)=_z+fijp1;
    vv(p) = 0.5/dz;
    p++; vi(p)=sij; vj(p)=_z+fijm1;
    vv(p) = -0.5/dz;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  // Reorder to create sparse matrix
  return comfi::util::syncSpMat(Avi, Avj, Avv,num_of_grid,DOF*num_of_grid); //reorder due to parallel construction
}

const sp_mat comfi::operators::buildBremove()
{
  const uint nnzp = 3;
  umat  Avi = zeros<umat>(nnzp, num_of_grid);
  umat  Avj = zeros<umat>(nnzp, num_of_grid);
  mat   Avv = zeros<mat> (nnzp, num_of_grid);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i
    const unsigned int  j=(index)/nx; //find j
    const unsigned int  ij=ind(i,j);      // find matrix index

    // Build sparse bulk vectors
    int p=-1.0;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=Bx+ij; vj(p)=Bx+ij;
    vv(p) = 1.0;
    p++; vi(p)=Bz+ij; vj(p)=Bz+ij;
    vv(p) = 1.0;
    p++; vi(p)=Bp+ij; vj(p)=Bp+ij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  // Reorder to create sparse matrix
  return arma::eye<arma::sp_mat>(num_of_elem, num_of_elem)-comfi::util::syncSpMat(Avi, Avj, Avv); //reorder due to parallel construction
}

const sp_mat comfi::operators::buildGLMremove()
{
  const uint nnzp = 1.0;
  umat  Avi = zeros<umat>(nnzp, num_of_grid);
  umat  Avj = zeros<umat>(nnzp, num_of_grid);
  mat   Avv = zeros<mat> (nnzp, num_of_grid);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    // indexing
    const unsigned int  i=(index)%nx; // find i
    const unsigned int  j=(index)/nx; //find j
    const unsigned int  ij=ind(i,j);      // find matrix index

    // Build sparse bulk vectors
    int p=-1;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=GLM+ij; vj(p)=GLM+ij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(index) = vi;
    Avj.col(index) = vj;
    Avv.col(index) = vv;
  }

  // Reorder to create sparse matrix
  return arma::eye<arma::sp_mat>(num_of_elem, num_of_elem)-comfi::util::syncSpMat(Avi, Avj, Avv); //reorder due to parallel construction
}

const vcl_sp_mat comfi::operators::bottomBz(const comfi::types::BgData &bg)
{
  const uint nnzp = 1.0;
  umat  Avi = zeros<umat>(nnzp, nx);
  umat  Avj = zeros<umat>(nnzp, nx);
  mat   Avv = zeros<mat> (nnzp, nx);

  #pragma omp parallel for schedule(static)
  for (uint i=0; i<nx; i++)
  {
    // indexing
    const unsigned int  ij=ind(i, 0);      // find matrix index

    // Build sparse bulk vectors
    int p=-1;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=Bz+ij; vj(p)=Bz+ij;
    //vv(p) = bg.BBz(i);
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(i) = vi;
    Avj.col(i) = vj;
    Avv.col(i) = vv;
  }
  // Reorder to create sparse matrix
  const sp_mat armaspmat = comfi::util::syncSpMat(Avi, Avj, Avv);//reorder due to parallel construction
  vcl_sp_mat ret;
  viennacl::copy(armaspmat, ret);
  return ret;
}

const vcl_sp_mat comfi::operators::bottomTp()
{
  const uint nnzp = 1;
  umat  Avi = zeros<umat>(nnzp, nx);
  umat  Avj = zeros<umat>(nnzp, nx);
  mat   Avv = zeros<mat> (nnzp, nx);

  #pragma omp parallel for schedule(static)
  for (uint i=0; i<nx; i++)
  {
    // indexing
    const unsigned int  ij=ind(i, 0);      // find matrix index

    // Build sparse bulk vectors
    int p=-1;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=E_p+ij; vj(p)=E_p+ij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(i) = vi;
    Avj.col(i) = vj;
    Avv.col(i) = vv;
  }
  // Reorder to create sparse matrix
  const sp_mat armaspmat = comfi::util::syncSpMat(Avi, Avj, Avv);//reorder due to parallel construction
  vcl_sp_mat ret;
  viennacl::copy(armaspmat, ret);
  return ret;
}

const vcl_sp_mat comfi::operators::bottomTn()
{
  const uint nnzp = 1;
  umat  Avi = zeros<umat>(nnzp, nx);
  umat  Avj = zeros<umat>(nnzp, nx);
  mat   Avv = zeros<mat> (nnzp, nx);

  #pragma omp parallel for schedule(static)
  for (uint i=0; i<nx; i++)
  {
    // indexing
    const unsigned int  ij=ind(i, 0);      // find matrix index

    // Build sparse bulk vectors
    int p = -1;
    uvec vi = zeros<uvec>(nnzp);
    uvec vj = zeros<uvec>(nnzp);
    vec  vv = zeros<vec> (nnzp);

    p++; vi(p)=E_n+ij; vj(p)=E_n+ij;
    vv(p) = 1.0;

    // Collect sparse values and locations
    Avi.col(i) = vi;
    Avj.col(i) = vj;
    Avv.col(i) = vv;
  }

  // Reorder to create sparse matrix
  const sp_mat armaspmat = comfi::util::syncSpMat(Avi, Avj, Avv);//reorder due to parallel construction
  vcl_sp_mat ret;
  viennacl::copy(armaspmat, ret);
  return ret;
}
