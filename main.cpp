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
  //Do initial setup
  const string starttime = comfi::util::gettimestr();
  stringstream logfilename;
  logfilename << "log/" << starttime << ".log";

  cout << "===Begin CoMFi Simulation===" << endl;
  #ifdef VIENNACL_WITH_OPENCL
      cout << "Device Info: " << endl << viennacl::ocl::current_device().info();
  #endif
  //cout << "OMP Max threads: " << omp_get_max_threads() << " OMP Devices: " << omp_get_default_device() << "/" << omp_get_num_devices() << endl;
  cout << "Parameters:" << endl;
  cout << "Grid size: " << nz << "x" << nx << " " << endl
       << "n_0: " << n_0 << " m^-3 | "
       << "L_0: " << l_0 << " m | "
       << "t_0: " << t_0 << " s | "
       << "V_0: " << V_0 << " m/s | "
       << "B_0: " << B_0/0.1e-3 << " G | "
       << "e_0: " << e_0 << " C | "
       << "T_0: " << T_0 << " K | "
       << "p_0: "<< p_0 << " Pa " << endl
       << "Width: " << width << " m | "
       << "Height: " << height << " m | "
       << "dx: " << dx*l_0 << " m | "
       << "dz: " << dz*l_0 << " m | "
       << "ds: " << ds*l_0 << " m " << endl;
  cout << "=============================================" << endl;
  stringstream logintromsg;
  logintromsg << "Params:\n"
              << "Grid size: " << nz << "x" << nx << endl;
  comfi::util::sendtolog(logintromsg.str(),logfilename.str());

    //Options
  uint   max_time_steps = 1000;
  double max_time = 1.0; //in t_0
  double tolerance = 1.e-6;   //gmres tolerance
  double start_dt = 1.e-8;   //start dt
  uint   save_every = 1;       //save solution every X time steps
  bool   errors = false;
  bool   resumed = false;   //If this run resumes a previous run
  //BoundaryCondition UpBC = NEUMANN, DownBC= NEUMANN, LeftBC=PERIODIC, RightBC=PERIODIC;


  if (argc != 1) // if arguments were passed
  {
    // parse arguments
    for(int i=1; i < argc;i++)
    {
      if(argv[i][0] == '-') // if starts with hyphen
      {
        string argument = argv[i];
        if (argument == "-max_time_steps")
        {
            string value = argv[i+1];
            if (i!=argc-1) {
              max_time_steps = stoi(value,nullptr,10);
              }
        }
        else if (argument == "-max_time")
        {
            string value = argv[i+1];
            if (i!=argc-1) { max_time = std::stod(value); }
        }else if (argument == "-save_every")
        {
            string value = argv[i+1];
            if (i!=argc-1) { save_every = stoi(value,nullptr,10); }
        }
        else if (argument == "-start_dt")
        {
            string value = argv[i+1];
            if (i!=argc-1) { start_dt = stoi(value,nullptr,10); }
        }
        else if (argument == "-tolerance")
        {
            string value = argv[i+1];
            if (i!=argc-1) { tolerance = std::stod(value); }
        }
        else if (argument == "-leftbc")
        {
            string value = argv[i+1];
            //if (i!=argc-1) { tolerance = stoi(value,nullptr,10); }
        }
        else if (argument == "-rightbc")
        {
            string value = argv[i+1];
            //if (i!=argc-1) { tolerance = stoi(value,nullptr,10); }
        }
        else if (argument == "-upbc")
        {
            string value = argv[i+1];
            //if (i!=argc-1) { tolerance = stoi(value,nullptr,10); }
        }
        else if (argument == "-downbc")
        {
            string value = argv[i+1];
            //if (i!=argc-1) { tolerance = stoi(value,nullptr,10); }
        }
        else if (argument == "-use_input_sol")
        {
            resumed = true;
        }
        else
        {
          cerr << "The command line option \'" << argument << "\' is not recognized. Using defaults for unspecified parameters." << endl;
        }
      }
    }
  }

  //Initiate
  viennacl::tools::timer vcl_timer[2]; // timers

  //auto init = mhdsim::util::calcInitialCondition(op);
  //auto init = mhdsim::util::calcReconnectionIC(op);
  //auto init = comfi::util::calcShockTubeIC(op);
  //auto init = comfi::util::calcSolerIC(op);
  //arma::vec x0 = std::get<0>(init);
  //const comfi::types::BgData bg;

  comfi::types::Context ctx(nx,
                            nz);
  ctx.set_dt(start_dt, false);
  unique_ptr<vcl_mat> xn_vcl(new vcl_mat(comfi::util::shock_tube_ic(ctx)));
  unique_ptr<vcl_mat> xn1_vcl(new vcl_mat(*xn_vcl));
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

  //Time Steps
  arma::uword start_step = 0;
  vcl_timer[0].start();
  // FIX LATER
  /*
  if (resumed)
  {
    xn.load("input/x0", arma::raw_binary);
    xn1.load("input/xn", arma::raw_binary);
    viennacl::fast_copy(xn, *xn_vcl);
    arma::vec last_t;
    last_t.load("input/t", arma::raw_binary);
    start_step = last_t.size();
    max_time_steps += last_t.size();
    time_elapsed = last_t(last_t.size()-1);
    std::string message = "Resuming simulation, starting steps at ";
    message += std::to_string(start_step);
    const double V = 1; //comfi::util::getmaxV(*xn_vcl, op);
    dt = 0.1*ds/V; // Update CFL
    comfi::util::sendtolog(message, logfilename.str());
  } else {
    comfi::util::saveSolution(*xn_vcl, 0, op); //save initial conditions
    const double V = 1; //comfi::util::getmaxV(*xn_vcl, op);
    dt = 0.1*ds/V; // Update CFL
    dt = 1.e-16;
  }
  */
  //Physical logging
  arma::vec t = arma::zeros<arma::vec>(max_time_steps); // record time elapsed in t_0
  arma::vec dt_n = arma::zeros<arma::vec>(max_time_steps); // record the changed timestep at step n
  arma::vec BE = arma::zeros<arma::vec>(max_time_steps); // record average magnetic energy
  arma::vec KE = arma::zeros<arma::vec>(max_time_steps); // record average kinetic energy
  arma::vec UE = arma::zeros<arma::vec>(max_time_steps); // record average internal energy
  arma::vec avgdivB = arma::zeros<arma::vec>(max_time_steps); // record average divB

  while ((ctx.time_step() < max_time_steps) && (ctx.time_elapsed() < max_time))
  {
    const uint relative_step = ctx.time_step() - start_step;
    double solve_time=0.0, build_time=0.0;
    //Output current parameters from last step
    // FIX LATER
    /*
    BE(relative_step) = comfi::util::getsumBE(*xn_vcl, op);
    KE(relative_step) = comfi::util::getsumKE(*xn_vcl, op);
    UE(relative_step) = comfi::util::getsumUE(*xn_vcl, op);
    cout << "Sum BE: " << BE(relative_step) << "\t\t" << "| Sum KE: " << KE(relative_step)
         << "\t| Sum UE: " << UE(relative_step) << endl << "E: " << UE(relative_step)+KE(relative_step)+BE(relative_step)
         << endl << endl;
         */

    //Update loop stuff
    t(relative_step) = ctx.time_elapsed();
    dt_n(relative_step) = ctx.dt();

    cout << ctx.time_step() << ": " ;
    vcl_timer[1].start();

    //const arma::sp_mat Ri_cpu = mhdsim::routines::computeRi(*xn_vcl, op);
    //static const arma::sp_mat one = arma::speye(num_of_elem, num_of_elem);
    //const arma::sp_mat LHS_cpu = one + Ri_cpu*dt; // Euler
    //vcl_sp_mat LHS;
    //viennacl::copy(LHS_cpu, LHS);

    //const vcl_vec RHS = mhdsim::routines::computeRHS_RK4(*xn_vcl, dt, time_elapsed, op, bg);

    build_time=vcl_timer[1].get();
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
    unique_ptr<vcl_mat> x0_vcl(new vcl_mat(comfi::routines::computeRHS_Euler(*xn_vcl, ctx)));
    //unique_ptr<vcl_mat> x0_vcl(new vcl_mat(*xn_vcl));

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

    // FIX LATER
    /*
    cout << "Del norm2: " << viennacl::linalg::norm_2(*x0_vcl - *xn_vcl);
    // DivB output
    const vcl_vec Bdirty = viennacl::linalg::prod(op.Bf, *x0_vcl);
    const vcl_vec divB = viennacl::linalg::prod(op.div, Bdirty);
    const double divBsum = viennacl::linalg::inner_prod(divB, divB);
    avgdivB(relative_step) = std::sqrt(divBsum)/num_of_grid;
    cout << " | Avg divB: " << avgdivB(relative_step) << endl;
    */

    const double V = comfi::util::getmaxV(*x0_vcl, ctx);
    cout << "Char speed: "  << V << " V_0" << endl;
    ctx.set_dt(0.8*0.5*ds/V);

    cout << "dt: " << ctx.dt() << " t_0";
    cout << "\t| Time: " << ctx.time_elapsed() << " t_0" << endl;

    //mhdsim::util::saveSolution(x0_vcl, -1, op);
    //mhdsim::util::saveSolution(xn_vcl, -2, op);
    //mhdsim::util::saveSolution(xn1_vcl, -3, op);

    //Save solution every save_every steps
    if ((ctx.time_step()%save_every) == 0)
    {
      comfi::util::saveSolution(*x0_vcl, ctx);
    }

    //Check whether steps should stop
    // FIX LATER
    /*
    if (sol_error > 1.e-3)
    {
      viennacl::fast_copy(RHS, RHSfinal);
      cout << "Stopped due to GMRES stalling (error too large)" << endl;
      comfi::util::sendtolog("GMRES Stalled error too large \n",logfilename.str());
      errors = true; break;
    }
    if (dt == 0.0 || !std::isfinite(dt))
    {
      viennacl::copy(RHS, RHSfinal);
      cout << "Stopped due to dt == 0.0 or nan" << endl;
      stringstream logmessage;
      logmessage << "Stopped due to dt == 0.0 or nan at step " << ctx.time_step() << endl;
      comfi::util::sendtolog(logmessage.str(),logfilename.str());
      errors = true; break;
    }
    */
    // FIX LATER
    /*
    if (!comfi::util::sanityCheck(*x0_vcl, op)) //not sane
    {
      cout << "Stopped due to negative pressure(s)." << endl;
      comfi::util::sendtolog("Stopped due to negative pressure(s).",logfilename.str());
      errors = true; //break;
    }
    */

    xn1_vcl = std::move(xn_vcl);
    xn_vcl = std::move(x0_vcl);
    // Save tracking parameters
    const arma::vec div_save = avgdivB(arma::span(0,relative_step));  div_save.save(arma::hdf5_name("output/mhdsim.h5", "divB", arma::hdf5_opts::replace));
    const arma::vec t_save = t(arma::span(0,relative_step));      t_save.save(arma::hdf5_name("output/mhdsim.h5", "t", arma::hdf5_opts::replace));
    const arma::vec dtn_save = dt_n(arma::span(0,relative_step)); dtn_save.save(arma::hdf5_name("output/mhdsim.h5", "dt", arma::hdf5_opts::replace));
    const arma::vec KE_save = KE(arma::span(0,relative_step));    KE_save.save(arma::hdf5_name("output/mhdsim.h5", "KE", arma::hdf5_opts::replace));
    const arma::vec BE_save = BE(arma::span(0,relative_step));    BE_save.save(arma::hdf5_name("output/mhdsim.h5", "BE", arma::hdf5_opts::replace));
    const arma::vec UE_save = UE(arma::span(0,relative_step));    UE_save.save(arma::hdf5_name("output/mhdsim.h5", "UE", arma::hdf5_opts::replace));
  }
  if (!errors) { comfi::util::sendtolog("Simulation finished without any errors.", logfilename.str()); }
  else         { comfi::util::sendtolog("Simulation aborted due to errors.", logfilename.str()); }

  cout << "Total exec time: " << vcl_timer[0].get() << endl;

  // Save last 3 sol
  //FIX LATER
  comfi::util::saveSolution(*xn_vcl, ctx, true);
  RHSfinal.save("output/RHSfinal", arma::raw_binary);

  return 0;
}
