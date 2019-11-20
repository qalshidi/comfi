#include <armadillo>
#include "comfi.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/linalg/maxmin.hpp"
#include "viennacl/forwards.h"

using namespace viennacl::linalg;

vcl_mat comfi::util::vec_to_mat(const vcl_vec &vec) {
  arma::vec cpu_vec(vec.size());
  viennacl::fast_copy(vec, cpu_vec);
  arma::mat cpu_mat(cpu_vec);
  vcl_mat return_mat(vec.size(), 1);
  viennacl::copy(cpu_mat, return_mat);
  return return_mat;
}

arma::sp_mat comfi::util::syncSpMat(const arma::umat Avi,
                                    const arma::umat Avj,
                                    const arma::mat Avv,
                                    const uint num_of_rows /*=num_of_elem*/,
                                    const uint num_of_cols /*=num_of_elem*/)
{
  arma::umat locations;
  arma::uvec elems = arma::find_finite(Avi);
  const arma::urowvec rowi = (Avi.elem(elems)).t();
  const arma::urowvec rowj = (Avj.elem(elems)).t();
  locations.insert_rows(0, rowi);
  locations.insert_rows(1, rowj);
  const arma::vec values = Avv.elem(elems);

  return arma::sp_mat(true, locations, values, num_of_rows, num_of_cols);
}

std::tuple<arma::vec, const comfi::types::BgData> comfi::util::calcSolerIC(const comfi::types::Operators &op) {
  arma::vec xn = arma::zeros<arma::vec>(num_of_elem);

  arma::vec z = arma::linspace(-10*l_0, 10*l_0, nz);
  arma::vec v = arma::cos(arma::datum::pi/20.0/l_0*z);

  #pragma omp parallel for schedule(static)
  for (uint index = 0; index < num_of_grid; index++) {
    //indexing
    const uint i=(index)%nx;
    const uint j=(index)/nx;
    const int ij=ind(i, j);
    xn(Vp+ij) = v(j);
    xn(Up+ij) = v(j)*2.0;
    xn(n_n+ij) = 2.0;
    xn(n_p+ij) = 1.0;
    xn(E_p+ij) = 1.0;
    xn(E_n+ij) = 1.0;
    xn(Bz+ij) = 1.0;

  }
  arma::vec b_z = arma::ones<arma::vec>(nx);
  arma::vec b_x = arma::zeros<arma::vec>(nz);
  arma::mat bn_p = arma::zeros<arma::mat>(nz, nx);
  arma::mat bn_n = arma::zeros<arma::mat>(nz, nx);
  const comfi::types::BgData bg(b_z, b_x, bn_p, bn_n);
  return std::make_tuple(xn, bg);
}

vcl_mat comfi::util::ot_vortex_ic(comfi::types::Context &ctx) {
  const double b = 1.0/std::sqrt(4.0*arma::datum::pi),
               n = 25.0/(36.0*arma::datum::pi),
               p = 5.0/(12.0*arma::datum::pi);

  arma::mat xn = arma::zeros<arma::mat>(ctx.num_of_grid(), ctx.num_of_eq());

  std::cout << "Building initial condition...";
  #pragma omp parallel for schedule(static)
  for (uint index = 0; index < ctx.num_of_grid(); index++) {
    //indexing
    const uint i=(index)%nx;
    const uint j=(index)/nx;
    const arma::uword ij = inds(i, j, ctx);

    xn(ij, n_p) = n;
    xn(ij, n_n) = n;
    xn(ij, E_p) = p/(gammamono-1.0);
    xn(ij, E_n) = p/(gammamono-1.0);
    xn(ij, Bp) = b;
    xn(ij, Bx) = -b*std::sin(2.0*arma::datum::pi*j*dz);
    xn(ij, Bz) = b*std::sin(4.0*arma::datum::pi*i*dx);
    xn(ij, Vx) = -1.0*n*std::sin(2.0*arma::datum::pi*j*dz);
    xn(ij, Vz) = n*std::sin(2.0*arma::datum::pi*i*dx);
  }
  std::cout << "Orszang-Tang Vortex Initial Conditions: (" << ctx.nx() << "," << ctx.nz() <<
               ") Size: " << xn.size() << std::endl;

  vcl_mat xn_vcl(ctx.num_of_grid(), ctx.num_of_eq());
  viennacl::copy(xn, xn_vcl);
  return xn_vcl;
}

vcl_mat comfi::util::shock_tube_ic(comfi::types::Context &ctx) {
  const double b = 0.75, b_l = 1.0, n_l = 1.0, p_l = 1.0, b_r = -1.0, n_r = 0.125, p_r = 0.1;

  arma::mat xn = arma::zeros<arma::mat>(ctx.num_of_grid(), ctx.num_of_eq());

  std::cout << "Building initial condition...";
  #pragma omp parallel for schedule(static)
  for (uint index = 0; index < ctx.num_of_grid(); index++) {
    //indexing
    const uint i=(index)%nx;
    const uint j=(index)/nx;
    const arma::uword ij = inds(i, j, ctx);

    if (j > nz/2) {
      xn(ij, n_p) = n_l;
      xn(ij, n_n) = n_l;
      xn(ij, E_n) = p_l/(gammamono-1.0);
      xn(ij, E_p) = p_l/(gammamono-1.0);
      xn(ij, Bx) = b_l;
      xn(ij, Bz) = b;
    } else {
      xn(ij, E_n) = p_r/(gammamono-1.0);
      xn(ij, E_p) = p_r/(gammamono-1.0);
      xn(ij, n_p) = n_r;
      xn(ij, n_n) = n_r;
      xn(ij, Bx) = b_r;
      xn(ij, Bz) = b;
    }
  }
  std::cout << "Sod\'s Shock Tube (" << ctx.nx() << "," << ctx.nz() <<
               ") Size: " << xn.size() << std::endl;

  vcl_mat xn_vcl(ctx.num_of_grid(), ctx.num_of_eq());
  viennacl::copy(xn, xn_vcl);
  return xn_vcl;
}

std::tuple<arma::vec, const comfi::types::BgData> comfi::util::shock_tube_ic(const comfi::types::Operators &op) {
  const double n_l = 1.0, p_l = 1.0, n_r = 0.125, p_r = 0.1;

  arma::vec xn = arma::zeros<arma::vec>(num_of_elem);

  #pragma omp parallel for schedule(static)
  for (uint index = 0; index < num_of_grid; index++) {
    //indexing
    const uint i=(index)%nx;
    const uint j=(index)/nx;
    const int ij=ind(i, j);

    if (j > nz/2) {
      xn(n_p+ij) = n_l;
      xn(n_n+ij) = n_l;
      xn(E_n+ij) = p_l/(gammamono-1.0);
      xn(E_p+ij) = p_l/n_l;
    } else {
      xn(E_n+ij) = p_r/(gammamono-1.0);
      xn(E_p+ij) = p_r/n_r;
      xn(n_p+ij) = n_r;
      xn(n_n+ij) = n_r;
    }
  }
  arma::vec b_z = arma::zeros<arma::vec>(nx);
  arma::vec b_x = arma::zeros<arma::vec>(nz);
  arma::mat bn_p = arma::zeros<arma::mat>(nz, nx);
  arma::mat bn_n = arma::zeros<arma::mat>(nz, nx);
  const comfi::types::BgData bg(b_z, b_x, bn_p, bn_n);
  return std::make_tuple(xn, bg);
}

std::tuple<arma::vec, const comfi::types::BgData> comfi::util::calcReconnectionIC(const comfi::types::Operators &op) {
  double x_n = 0.5*width; // Location of null point
  double z_n = 0.5*(height_start+height_end); // Location of null point
  arma::vec x = arma::linspace<arma::vec>(0, width, nx);
  arma::vec z = arma::linspace<arma::vec>(height_start, height_end, nz);
  arma::vec b_z = 0*arma::tanh((x-x_n*arma::ones<arma::vec>(nx))/l_0);
  arma::vec b_x = arma::tanh((z-z_n*arma::ones<arma::vec>(nz))/l_0);

  arma::mat bn_p = arma::zeros<arma::mat>(nz, nx);
  arma::mat bn_pz = arma::zeros<arma::vec>(nz);
  arma::mat bn_n = arma::zeros<arma::mat>(nz, nx);
  arma::mat bn_nz = arma::zeros<arma::vec>(nz);
  arma::mat temp = arma::zeros<arma::vec>(nz);

  arma::mat data; data.load("input/chromodata.csv");

  arma::interp1(data.col(1)*1000.0, data.col(5)*1.e6/n_0, z, bn_pz);
  arma::interp1(data.col(1)*1000.0, data.col(4)*1.e6/n_0, z, bn_nz);
  arma::interp1(data.col(1)*1000.0, data.col(2)/T_0, z, temp);

  arma::vec xn = arma::zeros<arma::vec>(num_of_elem);

  #pragma omp parallel for schedule(static)
  for (uint index = 0; index < num_of_grid; index++) {
    //indexing
    const uint i=(index)%nx;
    const uint j=(index)/nx;
    const int ij=ind(i, j);

    //const double BNpij = BNnij*mhdsim::sol::ionization_coeff(T0)/mhdsim::sol::recomb_coeff(T0);
    xn(n_p+ij) = bn_pz(j);
    bn_p(j, i) = bn_pz(j);
    xn(n_n+ij) = bn_nz(j);
    bn_n(j, i) = bn_nz(j);
    xn(E_p+ij) = temp(j);
    xn(E_n+ij) = temp(j);
    xn(Bz+ij) = 1.0;//b_z(i);
    xn(Bx+ij) = 0.0;//b_x(j);
  }

  const comfi::types::BgData bg(b_z, b_x, bn_p, bn_n);
  return std::make_tuple(xn, bg);
}

std::tuple<arma::vec, const comfi::types::BgData> comfi::util::calcInitialCondition(const comfi::types::Operators &op) {
  std::cout << "Calculating background magnetic field and densities..." << std::endl;
  arma::vec xn = arma::zeros<arma::vec>(num_of_elem);

#ifdef MHDSIM_BZTYPE_POISSON
#define MHDSIM_BZTYPE
  // make poisson for magnetic potential
  const uint nnzp = 6;
  arma::umat  Avi = arma::zeros<arma::umat>(nnzp, num_of_grid);
  arma::umat  Avj = arma::zeros<arma::umat>(nnzp, num_of_grid);
  arma::mat   Avv = arma::zeros<arma::mat> (nnzp, num_of_grid);
  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i = (index)%x_size;
    const unsigned int  j = (index)/x_size;
    const int           ij = inds(i,j);
    int 		ip1j = inds(i+1, j);
    int                 im1j = inds(i-1, j);
    int                 ijp1 = inds(i, j+1);
    int                 ijm1 = inds(i, j-1);

    if (i == x_size-1) { ip1j = ij; }
    if (i == 0) { im1j = ij; }
    //if (j == z_size-1) { ijp1 = ij; }

    int p=-1;
    arma::uvec vi = arma::zeros<arma::uvec>(nnzp);
    arma::uvec vj = arma::zeros<arma::uvec>(nnzp);
    arma::vec  vv = arma::zeros<arma::vec> (nnzp);

    p++; vi(p) = ij; vj(p) = ij;
    vv(p) = -2.0/dx2 - 2.0/dz2;
    if (j != z_size-1) {
    p++; vi(p) = ij; vj(p) = ijp1;
    vv(p) = 1.0/dz2; }
    if (j != 0) {
      p++; vi(p) = ij; vj(p) = ijm1;
      vv(p) = 1.0/dz2;
    }
    p++; vi(p) = ij; vj(p) = ip1j;
    vv(p) = 1.0/dx2 ;
    p++; vi(p) = ij; vj(p) = im1j;
    vv(p) = 1.0/dx2;

    // collect
    Avi(arma::span::all, index) = vi;
    Avj(arma::span::all, index) = vj;
    Avv(arma::span::all, index) = vv;
  }
  arma::sp_mat poisson = syncSpMat(Avi, Avj, Avv, num_of_grid, num_of_grid);
  vcl_sp_mat poisson_vcl(num_of_grid, num_of_grid);
  viennacl::copy(poisson, poisson_vcl);

  arma::vec b = arma::zeros<arma::vec>(num_of_grid);
  arma::vec x = dx * r_0 * arma::linspace(0, x_size-1, x_size);
  const double sigma = 2500.e3; // 10km
  arma::vec lower = arma::exp(-x%x/(sigma*sigma));
  arma::vec upper = 0.1 * arma::ones<arma::vec>(x_size);
  b(arma::span(0, x_size-1)) = lower / (dz2);
  b(arma::span(num_of_grid-x_size, num_of_grid-1)) = upper / (-dz2);
  vcl_vec b_vcl(num_of_grid);
  viennacl::fast_copy(b, b_vcl);

  viennacl::linalg::block_ilu_precond<vcl_sp_mat, viennacl::linalg::ilu0_tag> precond(poisson_vcl, viennacl::linalg::ilu0_tag(true));
  viennacl::linalg::gmres_tag my_gmres_tag(1.e-9, 100, 10);
  const vcl_vec scalar_pot = viennacl::linalg::solve(poisson_vcl, b_vcl, my_gmres_tag, precond);
  vcl_vec mag_field = prod(mhdsim::operators::field_scalarGrad(op.LeftBC, mhdsim::types::DIRICHLET, op.UpBC, op.RightBC), scalar_pot);
  const double norm = viennacl::linalg::max(mag_field);
  mag_field = mag_field / norm;
  const vcl_vec xnwB_vcl = prod(op.f2B, mag_field);
  const vcl_vec Bzij_vcl = prod(op.fdotz, mag_field);
  arma::vec Bzij(num_of_grid);
  viennacl::fast_copy(Bzij_vcl, Bzij);
  arma::vec xnwB(num_of_elem);
  viennacl::fast_copy(xnwB_vcl, xnwB);
  const arma::vec BBz = Bzij(arma::span(0, x_size-1));
#endif

#ifdef MHDSIM_BZTYPE_SQUARE
#define MHDSIM_BZTYPE
  int strongwidth = x_size/100; //km
  int dropwidth = x_size/100;
  int weakwidth = x_size - strongwidth - dropwidth;
  const arma::vec BBzstrong = arma::ones<arma::vec>(strongwidth);
  const arma::vec BBzdrop = arma::linspace<arma::vec>(1.0, 0.05, dropwidth);
  const arma::vec BBzweak = 0.05*arma::ones<arma::vec>(weakwidth);
  arma::vec BBz(x_size);
  BBz.subvec(0, strongwidth-1) = BBzstrong;
  BBz.subvec(strongwidth, strongwidth+dropwidth-1) = BBzdrop;
  BBz.subvec(strongwidth+dropwidth, x_size-1) = BBzweak;
  arma::vec xnwB = arma::zeros<arma::vec>(num_of_elem);
  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%x_size;
    const unsigned int  j=(index)/x_size;
    const int           ij=ind(i,j);
    xnwB(Bz+ij) = BBz(i);
  }
#endif

#ifndef MHDSIM_BZTYPE
#define MHDSIM_BZTYPE_EXP
#endif
#ifdef MHDSIM_BZTYPE_EXP
#define MHDSIM_BZTYPE
  double sigma = 50 * 1000; // 50 km
  double b_0 = 0.5; // 500 G
  const arma::vec x = arma::linspace<arma::vec>(0.0, width, nx);
  const arma::vec BBz = b_0*arma::exp(-(x%x)/(2*sigma*sigma));
  arma::vec xnwB = arma::zeros<arma::vec>(num_of_elem);
  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);
    xnwB(Bz+ij) = BBz(i);
  }
#endif

  arma::mat BNp = arma::zeros<arma::mat>(nz, nx);
  arma::mat BNn = arma::zeros<arma::mat>(nz, nx);
  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);

    const double 	Vi    = 0;//1.e-8*V_0;//1.e-6*V_0;
    //const double        BNpij = Np0*std::exp(-g*j*dz/T0);
    const double BNnij = std::exp(-g*j*dz/T0);
    const double BNpij = BNnij*comfi::sol::ionization_coeff(T0)/comfi::sol::recomb_coeff(T0);
    xn(n_p+ij) = BNpij;
    BNp(j, i) = BNpij;
    xn(n_n+ij) = BNnij;
    BNn(j, i) = BNnij;
    xn(E_p+ij) = T0;
    xn(E_n+ij) = T0;
    xn(Vz+ij)  = BNpij*Vi;
    xn(Uz+ij)  = BNnij*Vi;
  }

  std::cout << "done." << std::endl;
  const comfi::types::BgData bg(BBz, BNp, BNn);
  arma::vec x0 = xn + xnwB;
  return std::make_tuple(x0, bg);
}

arma::vec comfi::util::fillInitialCondition(const comfi::types::BgData &bg)
{
  arma::vec xn = arma::zeros<arma::vec>(num_of_elem);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<num_of_grid; index++)
  {
    //indexing
    const unsigned int  i=(index)%nx;
    const unsigned int  j=(index)/nx;
    const int           ij=ind(i,j);

    const double 	Vi    = 0;//1.e-8*V_0;//1.e-6*V_0;
    const double        BNpij = bg.BNp(j,i);
    const double        BNnij = bg.BNn(j,i);
    xn(n_p+ij) = BNpij;
    xn(n_n+ij) = BNnij;
    xn(E_p+ij) = T0;
    xn(E_n+ij) = T0;
    xn(Bz+ij)  = bg.BBz(i);
    xn(Vz+ij)  = BNpij*Vi;
    xn(Uz+ij)  = BNnij*Vi;
  }

  return xn;
}

std::string comfi::util::gettimestr()
{
      time_t rawtime;
      struct tm * timeinfo;
      char buffer[80];
      time (&rawtime);
      timeinfo = localtime(&rawtime);
      strftime(buffer,80,"%Y-%m-%d-%I-%M-%S",timeinfo);
      const std::string timestr(buffer);
      return timestr;
}

void comfi::util::sendtolog(const std::string message, const std::string filename)
{
    std::ofstream logfile(filename.c_str(), std::ios::app);
    if(logfile.is_open())
    {
        logfile << gettimestr() << ": " << message << std::endl;
    }
    else std::cout << "Could not open log file." << std::endl;

    logfile.close();
}

double comfi::util::getsumBE(const arma::vec &x0)
{
  arma::mat BE  = arma::zeros<arma::mat>(nz, nx);

  #pragma omp parallel for schedule(static)
  for(uint index=0; index<nx*nz; index++)
  {
	// indexing
  const unsigned int  n  =(index)%nx;
  const unsigned int  j  =(index)/nx;
	const int	    ij  =ind(n,j);

	const double Bxij = x0(Bx+ij);
	const double Bzij = x0(Bz+ij);
	const double Bpij = x0(Bp+ij);

	BE(j,n) = (Bxij*Bxij+Bzij*Bzij+Bpij*Bpij)/2.0;
  }

  return sum(sum(BE));
}

double comfi::util::getsumBE(const vcl_vec &x0, const comfi::types::Operators &op)
{
  const vcl_vec mag = viennacl::linalg::prod(op.Bf, x0);
  const vcl_vec be = 0.5*comfi::routines::dot_prod(mag, mag, op);
  return viennacl::linalg::sum(be);
}

double comfi::util::getsumKE(const arma::vec &x0)
{
  arma::mat KE = arma::zeros<arma::mat>(nz,nx);

  #pragma omp parallel for schedule(static)
  for(uint index=0; index<nx*nz; index++)
  {
	// indexing
  const unsigned int  n  =(index)%nx;
  const unsigned int  j  =(index)/nx;
	const int           ij =ind(n,j);

	const double Npij = x0(n_p+ij);
	const double Vxij = x0(Vx+ij);
	const double Vzij = x0(Vz+ij);
  const double Vpij = x0(Vp+ij);
	const double Nnij = x0(n_n+ij);
	const double Uxij = x0(Ux+ij);
	const double Uzij = x0(Uz+ij);
  const double Upij = x0(Up+ij);
  KE(j, n) = 0.5*(Vxij*Vxij+Vzij*Vzij+Vpij*Vpij)/Npij + 0.5*(Uxij*Uxij+Uzij*Uzij+Upij*Upij)/Nnij;
  }

  return sum(sum(KE));
}

double comfi::util::getsumKE(const vcl_vec &x0, const comfi::types::Operators &op)
{
  using namespace viennacl::linalg;
  const vcl_vec ionv = prod(op.Vf, x0);
  const vcl_vec neutv = prod(op.Uf, x0);
  vcl_vec ionke = 0.5*comfi::routines::dot_prod(ionv, ionv, op);
  ionke = viennacl::linalg::element_div(ionke, prod(op.Nps, x0));
  vcl_vec neutke = 0.5*comfi::routines::dot_prod(neutv, neutv, op);
  neutke = viennacl::linalg::element_div(neutke, prod(op.Nns, x0));
  return viennacl::linalg::sum(ionke+neutke);
}

double comfi::util::getsumUE(const arma::vec &x0)
{
  arma::mat E = arma::zeros<arma::mat>(nz,nx);

  #pragma omp parallel for schedule(static)
  for(uint index=0; index<nx*nz; index++)
  {
	// indexing
  const unsigned int  n  =(index)%nx; // find i
  const unsigned int  j  =(index)/nx;  //find j
	const int           ij =ind(n,j);

  const double UEpij = x0(n_p+ij)*x0(E_p+ij)/(gammamono-1.0);
  const double UEnij = x0(n_n+ij)*x0(E_n+ij)/(gammamono-1.0);
	E(j,n) = UEpij + UEnij;
  }

  return sum(sum(E));
}

double comfi::util::getsumUE(const vcl_vec &x0, const comfi::types::Operators &op)
{
  using namespace viennacl::linalg;
  const vcl_vec iue = element_prod(prod(op.Nps, x0), prod(op.Tps, x0))/(gammamono-1.0);
  const vcl_vec nue = element_prod(prod(op.Nns, x0), prod(op.Tns, x0))/(gammamono-1.0);
  return viennacl::linalg::sum(iue+nue);
}

bool comfi::util::saveSolution(const vcl_mat &x0, comfi::types::Context &ctx, const bool final)
{
  bool success = true;
  int timestep = ctx.time_step();
  if (final) { timestep = -1; }
  success *= saveScalar(viennacl::column(x0, n_p), "Np", timestep);
  success *= saveScalar(viennacl::column(x0, n_n), "Nn", timestep);
  success *= saveScalar(viennacl::column(x0, E_p), "Tp", timestep);
  success *= saveScalar(viennacl::column(x0, E_n), "Tn", timestep);
  success *= saveScalar(viennacl::column(x0, GLM), "GLM", timestep);
  success *= saveScalar(viennacl::column(x0, Vx), "NVx", timestep);
  success *= saveScalar(viennacl::column(x0, Vz), "NVz", timestep);
  success *= saveScalar(viennacl::column(x0, Vp), "NVp", timestep);
  success *= saveScalar(viennacl::column(x0, Ux), "NUx", timestep);
  success *= saveScalar(viennacl::column(x0, Uz), "NUz", timestep);
  success *= saveScalar(viennacl::column(x0, Up), "NUp", timestep);
  success *= saveScalar(viennacl::column(x0, Bx), "Bx", timestep);
  success *= saveScalar(viennacl::column(x0, Bz), "Bz", timestep);
  success *= saveScalar(viennacl::column(x0, Bp), "Bp", timestep);

  return success;
}

bool comfi::util::saveSolution(const vcl_vec &x0, const int &timestep, comfi::types::Operators &op)
{
  bool success = true;
  success *= saveField(prod(op.Bf, x0), "B", timestep);
  success *= saveField(prod(op.Vf, x0), "NV", timestep);
  success *= saveField(prod(op.Uf, x0), "NU", timestep);
  success *= saveScalar(prod(op.Nps, x0), "Np", timestep);
  success *= saveScalar(prod(op.Nns, x0), "Nn", timestep);
  success *= saveScalar(prod(op.Tps, x0), "Tp", timestep);
  success *= saveScalar(prod(op.Tns, x0), "Tn", timestep);
  success *= saveScalar(prod(op.GLMs, x0), "GLM", timestep);

  return success;
}

bool comfi::util::saveScalar(const arma::vec &x0, const std::string name, const int timestep)
{
  arma::mat sol(nz, nx);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<nx*nz;index++)
  {
    // indexing
    const unsigned int  n = (index)%nx;
    const unsigned int  j = nz-1 - (index)/nx;
    const int           sij = inds(n, j);

    sol(nz-1-j, n)  = x0(sij);
  }


  std::string folder = "output/";
  std::string filename = folder + "mhdsim.h5";
  std::string dataset = "/" + std::to_string(timestep) + "/" + name;
  sol.save(arma::hdf5_name(filename, dataset, arma::hdf5_opts::append+arma::hdf5_opts::trans));

  return true;
}

bool comfi::util::saveScalar(const vcl_vec &x0, const std::string name, const int timestep)
{
  arma::vec x0_cpu(x0.size());
  viennacl::fast_copy(x0,x0_cpu);
  bool rtn = comfi::util::saveScalar(x0_cpu, name, timestep);

  return rtn;
}

bool comfi::util::saveField(const arma::vec &x0, const std::string name, const int timestep)
{
  arma::mat xsol = arma::zeros<arma::mat>(nz, nx);
  arma::mat psol = arma::zeros<arma::mat>(nz, nx);
  arma::mat zsol = arma::zeros<arma::mat>(nz, nx);

  #pragma omp parallel for schedule(static)
  for (uint index=0; index<nx*nz;index++)
  {
    // indexing
    const unsigned int  n = (index)%nx;
    const unsigned int  j = (index)/nx;
    const int           fij= indf(n, j);

    xsol(nz-1-j, n)  = x0(_x+fij);
    zsol(nz-1-j, n)  = x0(_z+fij);
    psol(nz-1-j, n)  = x0(_p+fij);
  }

  std::string folder = "output/";
  std::string filename = folder + "mhdsim.h5";
  std::string dataset = "/" + std::to_string(timestep) + "/" + name + "x";
  xsol.save(arma::hdf5_name(filename, dataset, arma::hdf5_opts::append+arma::hdf5_opts::trans));

  dataset = "/" + std::to_string(timestep) + "/" + name + "z";
  zsol.save(arma::hdf5_name(filename, dataset, arma::hdf5_opts::append+arma::hdf5_opts::trans));

  dataset = "/" + std::to_string(timestep) + "/" + name + "p";
  psol.save(arma::hdf5_name(filename, dataset, arma::hdf5_opts::append+arma::hdf5_opts::trans));

  return true;
}

bool comfi::util::saveField(const vcl_vec &x0, const std::string name, const int timestep)
{
  arma::vec x0_cpu(x0.size()); viennacl::fast_copy(x0,x0_cpu); bool rtn = saveField(x0_cpu,name,timestep);

	return rtn;
}

bool comfi::util::sanityCheck(vcl_vec &xn, const comfi::types::Operators &op)
{
  bool saneP = true;
  bool saneN = true;
  // SCALARS
  const vcl_vec Np = prod(op.Nps, xn);
  const vcl_vec Nn = prod(op.Nns, xn);

  const vcl_vec Npabs = element_fabs(Np);
  const vcl_vec Npos = Np - Npabs;
  const double Npcheck = inner_prod(Npos,Npos);
  if (Npcheck != 0.0) // if not all pressures are positive
  {
    std::cout << "Np is negative" << std::endl;
    saneP = false;
    arma::vec P(Np.size()); viennacl::fast_copy(Np, P);
    #pragma omp parallel for schedule (static)
    for(uint index=0; index<nx*nz; index++)
    {
      if (P(index)<0.0)
      {
        const unsigned int  i=(index)%nx; // find i
        const unsigned int  j=(index)/nx; // find j
        std::cout << "Np = " << P(index) << " at (" << i << "," << j << ")" << std::endl;
        //P(index) = 1.e-6;
      }
    }
    //vcl_vec Npnew(P.size()); viennacl::fast_copy(P, Npnew);
    //const vcl_vec xnNp = viennacl::linalg::prod(op.s2Np, Np);
    //const vcl_vec xnNpnew = viennacl::linalg::prod(op.s2Np, Npnew);
    //xn = xn - xnNp + xnNpnew; // New energies with positive pressure
  }

  const vcl_vec Nnabs = element_fabs(Nn);
  const vcl_vec Nnpos = Nn - Nnabs;
  const double Nncheck = inner_prod(Nnpos, Nnpos);
  if (Nncheck != 0.0) // if not all pressures are positive
  {
    std::cout << "Nn is negative" << std::endl;
    saneN = false;
    arma::vec P(Nn.size()); viennacl::fast_copy(Nn, P);
    #pragma omp parallel for schedule (static)
    for(uint index=0; index<nx*nz; index++)
    {
      if (P(index) < 0.0)
      {
        const unsigned int  i=(index)%nx; // find i
        const unsigned int  j=(index)/nx; // find j
        std::cout << "Pn = " << P(index) << " at (" << i << "," << j << ")" << std::endl;
        //P(index) = 1.e-6;
      }
    }
    //vcl_vec Nnnew(P.size()); viennacl::fast_copy(P, Nnnew);
    //const vcl_vec xnNn = prod(op.s2Nn, Nn);
    //const vcl_vec xnNnnew = prod(op.s2Nn, Nnnew);
    //xn = xn - xnNn + xnNnnew; // New energies with positive pressure
  }

  return !((!saneP)|(!saneN)); // return false if either are insane (returns true if completely sane)
}

void comfi::util::interpret_arguments(comfi::util::Settings &settings, int argc, char** argv) {
  if (argc != 1) // if arguments were passed
  {
    // parse arguments
    for(int i=1; i < argc; i++)
    {
      if(argv[i][0] == '-') // if starts with hyphen
      {
        if (i != argc-1) {
          std::string argument = argv[i];
          if (argument == "-max_time_steps") {
            std::string value = argv[i+1];
            settings.max_time_steps = stoi(value,nullptr,10);
          } else if (argument == "-max_time") {
            std::string value = argv[i+1];
            settings.max_time = std::stod(value);
          } else if (argument == "-save_dt") {
            std::string value = argv[i+1];
            settings.save_dt = stoi(value,nullptr,10);
          } else if (argument == "-save_dn") {
            std::string value = argv[i+1];
            settings.save_dn = stoi(value,nullptr,10);
          } else if (argument == "-tolerance")
          {
            std::string value = argv[i+1];
            settings.tolerance = std::stod(value);
          } else if (argument == "-leftbc") {
            // TODO
            std::string value = argv[i+1];
          } else if (argument == "-rightbc") {
            // TODO
            std::string value = argv[i+1];
          } else if (argument == "-upbc") {
            // TODO
            std::string value = argv[i+1];
          } else if (argument == "-downbc") {
            // TODO
            std::string value = argv[i+1];
          } else if (argument == "-resume") {
            settings.resumed = true;
          } else {
            std::cerr << "The command line option \'" << argument << "\' is not recognized. Using defaults for unspecified parameters." << std::endl;
          }
        }
      }
    }
  }
}

double comfi::util::getmaxV(const vcl_mat &x0, comfi::types::Context &ctx) {
  using namespace viennacl::linalg;
  vcl_vec Np = viennacl::column(x0, n_p);
  vcl_vec Nn = viennacl::column(x0, n_n);
  vcl_vec NVx = viennacl::column(x0, Vx);
  vcl_vec NUx = viennacl::column(x0, Ux);
  vcl_vec NUz = viennacl::column(x0, Uz);
  vcl_vec NVz = viennacl::column(x0, Vz);
  const double local_p_x = viennacl::linalg::max(element_fabs(element_div(NVx, Np)));
  const double local_p_z = viennacl::linalg::max(element_fabs(element_div(NVz, Np)));
  const double local_n_x = viennacl::linalg::max(element_fabs(element_div(NUx, Nn)));
  const double local_n_z = viennacl::linalg::max(element_fabs(element_div(NUz, Nn)));
  std::cout << "Max local (p): (" << local_p_x << "," << local_p_z << ") | ";
  std::cout << "Max local (n): (" << local_n_x << "," << local_n_z << ")" << std::endl;
  const double fast_p_x = viennacl::linalg::max(comfi::routines::fast_speed_x(x0, ctx));
  const double fast_p_z = viennacl::linalg::max(comfi::routines::fast_speed_z(x0, ctx));
  std::cout << "Max fast (p): (" << fast_p_x << "," << fast_p_z << ") | ";
  const double sound_n_x = viennacl::linalg::max(comfi::routines::sound_speed_neutral(x0, ctx));
  const double sound_n_z = viennacl::linalg::max(comfi::routines::sound_speed_neutral(x0, ctx));
  std::cout << "Max sound speed (n): (" << sound_n_x << "," << sound_n_z << ")";
  std::cout << std::endl;

  /*
  return std::max(std::max(std::max(local_p_x, fast_p_x), std::max(local_p_z, fast_p_z)),
                  std::max(std::max(sound_n_x, local_n_x), std::max(sound_n_z, local_n_z)));
                  */
  return std::max(std::max(local_p_x+fast_p_x, local_p_z+fast_p_z),
                  std::max(sound_n_x+local_n_x, sound_n_z+local_n_z));
}

/*
vim: tabstop=2
vim: shiftwidth=2
vim: smarttab
vim: expandtab
*/
