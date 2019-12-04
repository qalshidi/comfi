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

vcl_mat comfi::util::ot_vortex_ic(comfi::types::Context &ctx) {
  const double b = 1.0/std::sqrt(4.0*arma::datum::pi),
               n = 25.0/(36.0*arma::datum::pi),
               p = 5.0/(12.0*arma::datum::pi);
  arma::mat xn = arma::zeros<arma::mat>(ctx.num_of_grid(), ctx.num_of_eq);

  std::cout << "Building initial condition...";
  #pragma omp parallel for collapse(2)
  for (arma::uword i=0; i<ctx.nx; i++) { for (arma::uword j=0; j<ctx.nz; j++) {
    //indexing
    const arma::uword ij = inds(i, j, ctx);

    xn(ij, ctx.n_p) = n;
    xn(ij, ctx.n_n) = n;
    xn(ij, ctx.E_p) = p/(ctx.gammamono-1.0);
    xn(ij, ctx.E_n) = p/(ctx.gammamono-1.0);
    //xn(ij, ctx.Bp) = b;
    xn(ij, ctx.Bx) = -b*std::sin(2.0*arma::datum::pi*j*ctx.dz);
    xn(ij, ctx.Bz) = b*std::sin(4.0*arma::datum::pi*i*ctx.dx);
    xn(ij, ctx.Vx) = -1.0*n*std::sin(2.0*arma::datum::pi*j*ctx.dz);
    xn(ij, ctx.Vz) = n*std::sin(2.0*arma::datum::pi*i*ctx.dx);
  }}

  std::cout << "Orszang-Tang Vortex Initial Conditions: (" << ctx.nx << "," << ctx.nz <<
               ") Size: " << xn.size() << std::endl;

  vcl_mat xn_vcl(xn.n_rows, xn.n_cols);
  viennacl::copy(xn, xn_vcl);
  return xn_vcl;
}

vcl_mat comfi::util::shock_tube_ic(comfi::types::Context &ctx) {
  const double b = 0.75, b_l = 1.0, n_l = 1.0, p_l = 1.0, b_r = -1.0, n_r = 0.125, p_r = 0.1;
  arma::mat xn = arma::zeros<arma::mat>(ctx.num_of_grid(), ctx.num_of_eq);

  std::cout << "Building initial condition...";
  #pragma omp parallel for collapse(2)
  for (uint i=0; i<ctx.nx; i++) { for (uint j=0; j<ctx.nz; j++) {
    const arma::uword ij = inds(i, j, ctx);

    if (j > ctx.nz/2) {
      xn(ij, ctx.n_p) = n_l;
      xn(ij, ctx.n_n) = n_l;
      xn(ij, ctx.E_n) = p_l/(ctx.gammamono-1.0);
      xn(ij, ctx.E_p) = p_l/(ctx.gammamono-1.0);
      xn(ij, ctx.Bx) = b_l;
      xn(ij, ctx.Bz) = b;
    } else {
      xn(ij, ctx.E_n) = p_r/(ctx.gammamono-1.0);
      xn(ij, ctx.E_p) = p_r/(ctx.gammamono-1.0);
      xn(ij, ctx.n_p) = n_r;
      xn(ij, ctx.n_n) = n_r;
      xn(ij, ctx.Bx) = b_r;
      xn(ij, ctx.Bz) = b;
    }
  }}
  std::cout << "Shock Tube (" << ctx.nx << "," << ctx.nz <<
               ") Size: " << xn.size() << std::endl;

  vcl_mat xn_vcl(xn.n_rows, xn.n_cols);
  viennacl::copy(xn, xn_vcl);
  return xn_vcl;
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

bool comfi::util::save_solution(const vcl_mat &x0,
                                comfi::types::Context &ctx,
                                const std::string &data_name) {
  bool success = true;
  int timestep = ctx.time_step();

  arma::mat savemat(x0.size1(), x0.size2());
  viennacl::copy(x0, savemat);
  static arma::mat gridmat(ctx.num_of_grid(), 2);
  static bool created = false;
  if (!created) {
    #pragma omp parallel for collapse(2)
    for (uint i=0; i<ctx.nx; i++) { for (uint j=0; j<ctx.nz; j++) {
      arma::uword ij = inds(i, j, ctx);
      gridmat(ij, 0) = i*ctx.dx*ctx.l_0;
      gridmat(ij, 1) = j*ctx.dz*ctx.l_0;
    }}
  }

  std::string folder = "output/";
  std::string filename = folder + "mhdsim.h5";
  std::string dataset = "/" + std::to_string(timestep) + "/" + data_name;
  std::string posset = "/" + std::to_string(timestep) + "/grid";
  savemat.save(arma::hdf5_name(filename, dataset, arma::hdf5_opts::append+arma::hdf5_opts::trans));
  gridmat.save(arma::hdf5_name(filename, posset, arma::hdf5_opts::append+arma::hdf5_opts::trans));

  return success;
}

void comfi::util::interpret_arguments(comfi::types::Settings &settings, int argc, char** argv) {
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
            settings.save_dt = std::atof(value.c_str());
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
          } else if (argument == "-restart") {
            settings.restart = true;
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
  const vcl_vec Np = viennacl::column(x0, ctx.n_p);
  const vcl_vec Nn = viennacl::column(x0, ctx.n_n);
  const vcl_vec NVx = viennacl::column(x0, ctx.Vx);
  const vcl_vec NUx = viennacl::column(x0, ctx.Ux);
  const vcl_vec NUz = viennacl::column(x0, ctx.Uz);
  const vcl_vec NVz = viennacl::column(x0, ctx.Vz);
  const vcl_vec local_p_x_vec = element_fabs(element_div(NVx, Np));
  const double local_p_x = viennacl::linalg::max(local_p_x_vec);
  const vcl_vec local_p_z_vec = element_fabs(element_div(NVz, Np));
  const double local_p_z = viennacl::linalg::max(local_p_z_vec);
  const vcl_vec local_n_x_vec = element_fabs(element_div(NUx, Nn));
  const double local_n_x = viennacl::linalg::max(local_n_x_vec);
  const vcl_vec local_n_z_vec = element_fabs(element_div(NUz, Nn));
  const double local_n_z = viennacl::linalg::max(local_n_z_vec);
  std::cout << "Max local (p): (" << local_p_x << "," << local_p_z << ") | ";
  std::cout << "Max local (n): (" << local_n_x << "," << local_n_z << ")" << std::endl;
  const double fast_p_x = viennacl::linalg::max(viennacl::column(comfi::routines::fast_speed_x(x0, ctx), 0));
  const double fast_p_z = viennacl::linalg::max(viennacl::column(comfi::routines::fast_speed_z(x0, ctx), 0));
  std::cout << "Max fast (p): (" << fast_p_x << "," << fast_p_z << ") | ";
  const double sound_n_x = viennacl::linalg::max(viennacl::column(comfi::routines::sound_speed_n(x0, ctx), 0));
  const double sound_n_z = viennacl::linalg::max(viennacl::column(comfi::routines::sound_speed_n(x0, ctx), 0));
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
