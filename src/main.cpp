// std includes
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>
#include <memory>

// Armadillo includes and preprocessors
#include <armadillo>
// ViennaCL includes
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/ilu.hpp"
//#include "viennacl/linalg/ilu/chow_patel_ilu.hpp"
#include "viennacl/linalg/lu.hpp"
#include "viennacl/linalg/amg.hpp"
#include "viennacl/linalg/sum.hpp"
#include "viennacl/linalg/maxmin.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/forwards.h"

// Main CoMFi include
#include "comfi.hpp"

using namespace std;

int main(int argc, char** argv)
{

  // Create intermediate variable but use constant for safety
	comfi::util::Settings __set;
	if (argc > 1) {
		comfi::util::interpret_arguments(__set, argc, argv);
	}
  const comfi::util::Settings settings = __set;

  // Initial std output
  cout << "Begin CoMFi Simulation" << endl;
  cout << "======================" << endl 
       << endl;
  #ifdef VIENNACL_WITH_OPENCL
    cout << "Device Info:" << endl 
         << "------------" << endl 
         << viennacl::ocl::current_device().info()
         << endl;
  #endif
  #ifdef _OPENMP
    cout << "OpenMP Info:" << endl
         << "------------" << endl
         << "- OMP Max threads: " << omp_get_max_threads() << endl
         << "- OMP Devices: " << omp_get_default_device() << "/" << omp_get_num_devices() << endl
         << endl;
  #endif
  cout << "Parameters:" << endl;
  cout << "-----------" << endl;
  cout << "- Grid size: (" << nz << ", " << nx << ") " << endl
       << "- Normalization constants:" << endl
       << " - " << "n_0: " << n_0 << " m^-3" << endl
       << " - " << "L_0: " << l_0 << " m" << endl
       << " - " << "t_0: " << t_0 << " s" << endl
       << " - " << "V_0: " << V_0 << " m/s" << endl
       << " - " << "B_0: " << B_0/0.1e-3 << " G" << endl
       << " - " << "e_0: " << e_0 << " C" << endl
       << " - " << "T_0: " << T_0 << " K" << endl
       << " - " << "p_0: "<< p_0 << " Pa" << endl
       << " - " << "Width: " << width << " m" << endl
       << " - " << "Height: " << height << " m" << endl
       << " - " << "dx: " << dx*l_0 << " m | " << "dz: " << dz*l_0 << " m | " << "ds: " << ds*l_0 << " m " << endl
       << endl;

  //Initiate
  viennacl::tools::timer vcl_timer[2]; // timers

  //auto init = mhdsim::util::calcInitialCondition(op);
  //auto init = mhdsim::util::calcReconnectionIC(op);
  //auto init = comfi::util::calcShockTubeIC(op);
  //auto init = comfi::util::calcSolerIC(op);
  //arma::vec x0 = std::get<0>(init);
  //const comfi::types::BgData bg;

  comfi::types::Context ctx(nx,
                            nz,
                            comfi::types::PERIODIC,
                            comfi::types::PERIODIC,
                            comfi::types::PERIODIC,
                            comfi::types::PERIODIC);
  //unique_ptr<vcl_mat> xn_vcl(new vcl_mat(comfi::util::shock_tube_ic(ctx)));
  unique_ptr<vcl_mat> xn_vcl(new vcl_mat(comfi::util::ot_vortex_ic(ctx)));
  unique_ptr<vcl_mat> xn1_vcl(new vcl_mat(*xn_vcl));
  comfi::util::saveSolution(*xn_vcl, ctx);
  arma::mat RHSfinal = arma::zeros<arma::mat>(xn_vcl->size1(), xn_vcl->size2());
  cout << "Compiling kernels ... ";
  std::ifstream phi_file("kernels_ocl/phi.c");
  string phi_code((std::istreambuf_iterator<char>(phi_file)),
                   std::istreambuf_iterator<char>()
                 );
  viennacl::ocl::program & phi_prog = viennacl::ocl::current_context().add_program(phi_code.c_str(), "fluxl"); // compile flux opencl kernel
  std::ifstream eig_file("kernels_ocl/element_max.c");
  string eig_code((std::istreambuf_iterator<char>(eig_file)),
                   std::istreambuf_iterator<char>()
                 );
  viennacl::ocl::program & eig_prog = viennacl::ocl::current_context().add_program(eig_code.c_str(), "element_max"); // compile eigenvalue opencl kernel
  cout << "built." << endl;

  // Full execution timer
  vcl_timer[0].start();
  
  //Physical logging
  arma::vec t = arma::zeros<arma::vec>(settings.max_time_steps); // record time elapsed in t_0
  arma::vec dt_n = arma::zeros<arma::vec>(settings.max_time_steps); // record the changed timestep at step n
  arma::vec BE = arma::zeros<arma::vec>(settings.max_time_steps); // record average magnetic energy
  arma::vec KE = arma::zeros<arma::vec>(settings.max_time_steps); // record average kinetic energy
  arma::vec UE = arma::zeros<arma::vec>(settings.max_time_steps); // record average internal energy
  arma::vec avgdivB = arma::zeros<arma::vec>(settings.max_time_steps); // record average divB

  // Begin solving/advancing
  while (((ctx.time_step() < settings.max_time_steps) || (settings.max_time_steps < 0)) &&
         ((ctx.time_elapsed() < settings.max_time) || (settings.max_time < 0.0))) {

    double solve_time=0.0, build_time=0.0;

    // Figure out time step
    const double V = comfi::util::getmaxV(*xn_vcl, ctx);
    cout << "Char speed: "  << V << " V_0" << endl;
    ctx.set_dt(0.8*0.5*ds/V);
    cout << "dt: " << ctx.dt() << " t_0";
    cout << "\t| Time: " << ctx.time_elapsed() << " t_0" << endl;

    //Update loop stuff
    t(ctx.time_step()) = ctx.time_elapsed();
    dt_n(ctx.time_step()) = ctx.dt();

    cout << ctx.time_step() << ": " ;
    vcl_timer[1].start();

    //const arma::sp_mat Ri_cpu = mhdsim::routines::computeRi(*xn_vcl, op);
    //static const arma::sp_mat one = arma::speye(num_of_elem, num_of_elem);
    //const arma::sp_mat LHS_cpu = one + Ri_cpu*dt; // Euler
    //vcl_sp_mat LHS;
    //viennacl::copy(LHS_cpu, LHS);

    //const vcl_vec RHS = mhdsim::routines::computeRHS_RK4(*xn_vcl, dt, time_elapsed, op, bg);
    build_time = vcl_timer[1].get();
    
    //GMRES
    // solve (e.g. using GMRES solver)
    vcl_timer[1].start();
    // create and compute preconditioner:
    //viennacl::linalg::ilu0_tag ilu0_config(true);
    //viennacl::linalg::block_ilu_precond<vcl_sp_mat, viennacl::linalg::ilu0_tag> vcl_precond(LHS, ilu0_config);
    //viennacl::linalg::ilu0_precond< vcl_sp_mat > vcl_precond(LHS, viennacl::linalg::ilu0_tag());
    //viennacl::linalg::gmres_tag my_solver_tag(tolerance, 100, 20);
    //viennacl::linalg::bicgstab_tag my_solver_tag(tolerance, 100, 20);
    //unique_ptr<vcl_vec> x0_vcl(new vcl_vec(viennacl::linalg::solve(LHS, RHS, my_solver_tag, vcl_precond)));
    //unique_ptr<vcl_mat> x0_vcl(new vcl_mat(comfi::routines::computeRHS_Euler(*xn_vcl, ctx)));
    unique_ptr<vcl_mat> x0_vcl(new vcl_mat(comfi::routines::computeRHS_RK4(*xn_vcl, ctx)));
    //unique_ptr<vcl_mat> x0_vcl(new vcl_mat(*xn_vcl));
    ctx.advance();

    //arma::vec diag_LHS(LHS_cpu.diag());
    //vcl_vec new_LHS(num_of_elem);
    //viennacl::fast_copy(diag_LHS, new_LHS);
    //unique_ptr<vcl_vec> x0_vcl(new vcl_vec(viennacl::linalg::element_div(RHS, new_LHS)));

    solve_time = vcl_timer[1].get();

//    cout << " GMRES(" << my_solver_tag.iters();
//    cout << "," << my_solver_tag.error();
//    cout << ") | Build Time:" << build_time;
//    cout << "s\t| Sol Time:" << solve_time << "s" << endl;
//    const double sol_error = my_solver_tag.error();
    double sol_error = 0;

    //mhdsim::util::saveSolution(x0_vcl, -1, op);
    //mhdsim::util::saveSolution(xn_vcl, -2, op);
    //mhdsim::util::saveSolution(xn1_vcl, -3, op);

    //Save solution every save_dt time
    static double time_since_last_save = ctx.time_elapsed();
    if ((time_since_last_save > settings.save_dt) && (settings.save_dt > 0.0)) {
      comfi::util::saveSolution(*x0_vcl, ctx);
      time_since_last_save = 0.0;
    }

    //Save solution every save_dn steps
    if ((ctx.time_step()%settings.save_dn) == 0) {
      comfi::util::saveSolution(*x0_vcl, ctx);
    }

    xn1_vcl = std::move(xn_vcl);
    xn_vcl = std::move(x0_vcl);
    // Save tracking parameters
    const arma::vec div_save = avgdivB(arma::span(0,ctx.time_step()-1));  div_save.save(arma::hdf5_name("output/mhdsim.h5", "divB", arma::hdf5_opts::replace));
    const arma::vec t_save = t(arma::span(0,ctx.time_step()-1));      t_save.save(arma::hdf5_name("output/mhdsim.h5", "t", arma::hdf5_opts::replace));
    const arma::vec dtn_save = dt_n(arma::span(0,ctx.time_step()-1)); dtn_save.save(arma::hdf5_name("output/mhdsim.h5", "dt", arma::hdf5_opts::replace));
    const arma::vec KE_save = KE(arma::span(0,ctx.time_step()-1));    KE_save.save(arma::hdf5_name("output/mhdsim.h5", "KE", arma::hdf5_opts::replace));
    const arma::vec BE_save = BE(arma::span(0,ctx.time_step()-1));    BE_save.save(arma::hdf5_name("output/mhdsim.h5", "BE", arma::hdf5_opts::replace));
    const arma::vec UE_save = UE(arma::span(0,ctx.time_step()-1));    UE_save.save(arma::hdf5_name("output/mhdsim.h5", "UE", arma::hdf5_opts::replace));
  }

  cout << "Total exec time: " << vcl_timer[0].get() << endl;

  // Save last 3 sol
  //FIX LATER
  comfi::util::saveSolution(*xn_vcl, ctx, true);
  RHSfinal.save("output/RHSfinal", arma::raw_binary);

  return 0;
}

/*
vim: tabstop=2
vim: shiftwidth=2
vim: smarttab
vim: expandtab
*/
