#include "comfi.h"

using namespace viennacl::linalg;

comfi::types::Operators::Operators(const BoundaryCondition _LeftBC /*=MIRROR*/,
                const BoundaryCondition _RightBC /*MIRROR*/,
                const BoundaryCondition _UpBC /*NEUMANN*/,
                const BoundaryCondition _DownBC /*NEUMANN*/
                ) : LeftBC(_LeftBC), RightBC(_RightBC), UpBC(_UpBC), DownBC(_DownBC)
{
  viennacl::copy(comfi::operators::buildPjp1(UpBC),Pjp1);
  viennacl::copy(comfi::operators::buildPjm1(DownBC),Pjm1);
  viennacl::copy(comfi::operators::buildPip1(RightBC),Pip1);
  viennacl::copy(comfi::operators::buildPim1(LeftBC),Pim1);
  viennacl::copy(comfi::operators::buildPip2(RightBC),Pip2);
  viennacl::copy(comfi::operators::buildPim2(LeftBC),Pim2);
  viennacl::copy(comfi::operators::buildPjp2(UpBC),Pjp2);
  viennacl::copy(comfi::operators::buildPjm2(DownBC),Pjm2);
  viennacl::copy(comfi::operators::buildSG(),SG);
  viennacl::copy(comfi::operators::buildPEigVx(),PEVx);
  viennacl::copy(comfi::operators::buildPEigVz(),PEVz);
  viennacl::copy(comfi::operators::buildPN(),PN);
  viennacl::copy(comfi::operators::buildBfield(),Bf);
  viennacl::copy(comfi::operators::buildpFBB(),pFBB);
  viennacl::copy(comfi::operators::buildFxVB(),FxVB);
  viennacl::copy(comfi::operators::buildFzVB(),FzVB);
  viennacl::copy(comfi::operators::buildVfield(),Vf);
  viennacl::copy(comfi::operators::buildUfield(),Uf);
  viennacl::copy(comfi::operators::field_xProjection(),fdotx);
  viennacl::copy(comfi::operators::field_pProjection(),fdotp);
  viennacl::copy(comfi::operators::field_zProjection(),fdotz);
  viennacl::copy(comfi::operators::field_scalar2field(),s2f);
  viennacl::copy(comfi::operators::field_scalar2xfield(),s2xf);
  viennacl::copy(comfi::operators::field_scalar2pfield(),s2pf);
  viennacl::copy(comfi::operators::field_scalar2zfield(),s2zf);
  viennacl::copy(comfi::operators::field_field2scalar(),f2s);
  viennacl::copy(comfi::operators::buildCross1(),cross1);
  viennacl::copy(comfi::operators::buildCross2(),cross2);
  viennacl::copy(comfi::operators::buildCurl(LeftBC,RightBC,UpBC,DownBC),curl);
  viennacl::copy(comfi::operators::builds2GLM(),s2GLM);
  viennacl::copy(comfi::operators::builds2Np(),s2Np);
  viennacl::copy(comfi::operators::builds2Nn(),s2Nn);
  viennacl::copy(comfi::operators::builds2Tp(),s2Tp);
  viennacl::copy(comfi::operators::builds2Tn(),s2Tn);
  viennacl::copy(comfi::operators::builds2Vx(),s2Vx);
  viennacl::copy(comfi::operators::builds2Vz(),s2Vz);
  viennacl::copy(comfi::operators::builds2Ux(),s2Ux);
  viennacl::copy(comfi::operators::builds2Uz(),s2Uz);
  viennacl::copy(comfi::operators::builds2Bx(),s2Bx);
  viennacl::copy(comfi::operators::builds2Bz(),s2Bz);
  viennacl::copy(comfi::operators::buildGLMscalar(),GLMs);
  viennacl::copy(comfi::operators::buildNpscalar(),Nps);
  viennacl::copy(comfi::operators::buildNnscalar(),Nns);
  viennacl::copy(comfi::operators::buildTpscalar(),Tps);
  viennacl::copy(comfi::operators::buildTnscalar(),Tns);
  viennacl::copy(comfi::operators::buildf2B(),f2B);
  viennacl::copy(comfi::operators::buildf2V(),f2V);
  viennacl::copy(comfi::operators::buildf2U(),f2U);
  viennacl::copy(comfi::operators::field_scalarGrad(LeftBC, DownBC, UpBC, RightBC), grad);
  viennacl::copy(comfi::operators::field_fieldDiv(LeftBC,RightBC,UpBC,DownBC),div);
  viennacl::copy(comfi::operators::buildBottomBC(),BottomBC);
  const arma::sp_mat I_cpu = arma::eye<arma::sp_mat>(num_of_elem, num_of_elem);
  viennacl::copy(I_cpu, I);
  const arma::sp_mat ImB_cpu = I_cpu-comfi::operators::buildBottomBC();
  viennacl::copy(ImB_cpu, ImBottom);
  const arma::sp_mat ImT_cpu = I_cpu-comfi::operators::buildTopBC();
  viennacl::copy(ImT_cpu, ImTop);
  const arma::sp_mat ImGLM_cpu = comfi::operators::buildGLMremove();
  viennacl::copy(ImGLM_cpu, ImGLM);
};

comfi::types::BoundaryCondition comfi::types::Operators::getLeftBC() const { return LeftBC; }
comfi::types::BoundaryCondition comfi::types::Operators::getRightBC() const { return RightBC; }
comfi::types::BoundaryCondition comfi::types::Operators::getUpBC() const { return UpBC; }
comfi::types::BoundaryCondition comfi::types::Operators::getDownBC() const { return DownBC; }

comfi::types::BgData::BgData(std::string BBzfile /*= "input/BBz.csv"*/,
                              std::string BNpfile /*= "input/BNp.csv"*/,
                              std::string BNnfile /*= "input/BNn.csv"*/)
{
  BBz.load(BBzfile);
  BNp.load(BNpfile);
  BNn.load(BNnfile);
}
