#include "polyscope/polyscope.h"

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

#include <iostream>
#include <unordered_set>
#include <utility>
#include <deque>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "RigidBodyInstance.h"
#include "RigidBodyTemplate.h"
#include "VectorMath.h"
#include <fstream>
#include "misc/cpp/imgui_stdlib.h"

bool running_;
double time_;
SimParameters params_;
std::string sceneFile_;

std::vector<RigidBodyTemplate*> templates_;
std::vector<RigidBodyInstance*> bodies_;

Eigen::MatrixXd renderQ;
Eigen::MatrixXi renderF;

void printv3d(Eigen::Vector3d v) {
    std::cout << v[0] << ", " << v[1] << ", " << v[2] << std::endl;
}

void updateRenderGeometry()
{
    int totverts = 0;
    int totfaces = 0;
    for (RigidBodyInstance* rbi : bodies_)
    {
        totverts += rbi->getTemplate().getVerts().rows();
        totfaces += rbi->getTemplate().getFaces().rows();
    }
    renderQ.resize(totverts, 3);
    renderF.resize(totfaces, 3);
    int voffset = 0;
    int foffset = 0;
    for (RigidBodyInstance* rbi : bodies_)
    {
        int nverts = rbi->getTemplate().getVerts().rows();
        for (int i = 0; i < nverts; i++)
            renderQ.row(voffset + i) = (rbi->c + VectorMath::rotationMatrix(rbi->theta) * rbi->getTemplate().getVerts().row(i).transpose()).transpose();
        int nfaces = rbi->getTemplate().getFaces().rows();
        for (int i = 0; i < nfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                renderF(foffset + i, j) = rbi->getTemplate().getFaces()(i, j) + voffset;
            }
        }
        voffset += nverts;
        foffset += nfaces;
    }
}

void loadScene()
{
    for (RigidBodyInstance* rbi : bodies_)
        delete rbi;
    for (RigidBodyTemplate* rbt : templates_)
        delete rbt;
    bodies_.clear();
    templates_.clear();

    std::string prefix;
    std::string scenefname = std::string("scenes/") + sceneFile_;
    std::ifstream ifs(scenefname);
    if (!ifs)
    {
        // run from the build directory?
        prefix = "../";
        std::string fullname = prefix + scenefname;
        ifs.open(fullname);
        if (!ifs)
        {
            prefix = "../../";
            fullname = prefix + scenefname;
            ifs.open(fullname);
            if (!ifs)
                return;
        }
    }


    int nbodies;
    ifs >> nbodies;
    for (int body = 0; body < nbodies; body++)
    {
        std::string meshname;
        ifs >> meshname;
        meshname = prefix + std::string("meshes/") + meshname;
        double scale;
        ifs >> scale;
        RigidBodyTemplate* rbt = new RigidBodyTemplate(meshname, scale);
        double rho;
        ifs >> rho;
        Eigen::Vector3d c, theta, cvel, w;
        for (int i = 0; i < 3; i++)
            ifs >> c[i];
        for (int i = 0; i < 3; i++)
            ifs >> theta[i];
        for (int i = 0; i < 3; i++)
            ifs >> cvel[i];
        for (int i = 0; i < 3; i++)
            ifs >> w[i];
        RigidBodyInstance* rbi = new RigidBodyInstance(*rbt, c, theta, cvel, w, rho);
        templates_.push_back(rbt);
        bodies_.push_back(rbi);
    }
}

void initSimulation()
{
    time_ = 0;
    loadScene();
    updateRenderGeometry();
}

void build_dofs(Eigen::VectorXd& q, Eigen::VectorXd& qdot) {
    for (int i = 0; i < bodies_.size(); i++) {
        q.segment<3>(i * 3 + 0) = bodies_[i]->c;
        q.segment<3>(i * 3 + bodies_.size() * 3) = bodies_[i]->theta;
        qdot.segment<3>(i * 3 + 0) = bodies_[i]->cvel;
        qdot.segment<3>(i * 3 + bodies_.size() * 3) = bodies_[i]->w;
    }
}

void unbuild_dofs(Eigen::VectorXd q, Eigen::VectorXd qdot) {
    for (int i = 0; i < bodies_.size(); i++) {
        bodies_[i]->c = q.segment<3>(i * 3 + 0);
        bodies_[i]->theta = q.segment<3>(i * 3 + bodies_.size() * 3);
        bodies_[i]->cvel = qdot.segment<3>(i * 3 + 0);
        bodies_[i]->w = qdot.segment<3>(i * 3 + bodies_.size() * 3);
    }
}

void computeMassInverse(Eigen::SparseMatrix<double>& Minv)
{
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < bodies_.size(); i++) {
        triplets.emplace_back(i * 3 + 0, i * 3 + 0,
            1.0 / bodies_[i]->density * bodies_[i]->getTemplate().getVolume());
        triplets.emplace_back(i * 3 + 1, i * 3 + 1,
            1.0 / bodies_[i]->density * bodies_[i]->getTemplate().getVolume());
        triplets.emplace_back(i * 3 + 2, i * 3 + 2,
            1.0 / bodies_[i]->density * bodies_[i]->getTemplate().getVolume());
    }
    Minv.setFromTriplets(triplets.begin(), triplets.end());
}

void computeTranslationalForceAndHessian(const Eigen::VectorXd& q, const Eigen::VectorXd& qprev, Eigen::VectorXd& F, Eigen::SparseMatrix<double>& H, double h)
{

}
void computeRotationalForceAndHessian(const Eigen::VectorXd& q, const Eigen::VectorXd& qprev, Eigen::VectorXd& F, Eigen::SparseMatrix<double>& H, double h)
{

}

Eigen::Vector3d residualOmega_i(
    const Eigen::Vector3d& theta,
    const Eigen::Vector3d& omega_i,
    const Eigen::Vector3d& omega,
    const Eigen::Matrix3d& MI,
    double rho,
    double h)
{
    Eigen::Vector3d prev =
        VectorMath::TMatrix(h * omega_i).inverse().transpose()
        * MI * omega_i;

    Eigen::Vector3d next =
        VectorMath::rotationMatrix(h * omega)
        * VectorMath::TMatrix(h * omega).inverse().transpose()
        * MI * omega;

    return (rho / h)
        * VectorMath::TMatrix(theta).transpose()
        * (prev - next);
}

void compute_residualOmega(
    Eigen::VectorXd& R,
    const Eigen::VectorXd& q_r,
    const Eigen::VectorXd& qdot_r,
    const Eigen::VectorXd& qdot_rp1,
    const Eigen::VectorXd& F
) {
    R = F;
    for (int i = 0; i < bodies_.size(); i++) {
        R.segment<3>(i * 3) +=
            residualOmega_i(
                q_r.segment<3>(i * 3),
                qdot_r.segment<3>(i * 3),
                qdot_rp1.segment<3>(i * 3),
                bodies_[i]->getTemplate().getInertiaTensor(),
                bodies_[i]->density,
                params_.timeStep
            );
    }
}

Eigen::Matrix3d approxJacobianOmega_i(
    const Eigen::Vector3d& theta,
    const Eigen::Vector3d& omega,
    const Eigen::Matrix3d& MI,
    double rho,
    double h)
{
    return -(rho / h)
        * VectorMath::TMatrix(theta).transpose()
        * VectorMath::rotationMatrix(h * omega)
        * VectorMath::TMatrix(h * omega).inverse().transpose()
        * MI;
}

void compute_deltaOmega(
    Eigen::VectorXd& delta,
    const Eigen::VectorXd& R,
    const Eigen::VectorXd& q_r,
    const Eigen::VectorXd& qdot_rp1
) {
    delta.resize(3 * bodies_.size());

    for (int i = 0; i < bodies_.size(); i++) {
        int base = i * 3;

        Eigen::Matrix3d Ji = approxJacobianOmega_i(
            q_r.segment<3>(base),
            qdot_rp1.segment<3>(base),
            bodies_[i]->getTemplate().getInertiaTensor(),
            bodies_[i]->density,
            params_.timeStep
        );

        Eigen::Vector3d Ri = R.segment<3>(base);

        Eigen::Vector3d deltai = Ji.fullPivLu().solve(-Ri);

        if (!deltai.allFinite()) {
            assert(false);
        }

        delta.segment<3>(base) = deltai;
    }
}


void numericalIntegration(Eigen::VectorXd& q, Eigen::VectorXd& qdot) {
    auto h = params_.timeStep;
    Eigen::SparseMatrix<double> Minv(bodies_.size() * 3, bodies_.size() * 3);
    computeMassInverse(Minv);

    {
        // translational forces
        Eigen::VectorXd q_t = q.segment(0, bodies_.size() * 3);
        Eigen::VectorXd qdot_t = qdot.segment(0, bodies_.size() * 3);
        Eigen::VectorXd qprev_t = q_t;
        Eigen::VectorXd F(3* bodies_.size());
        Eigen::SparseMatrix<double> H(3 * bodies_.size(), 3 * bodies_.size());
        Eigen::SparseMatrix<double> I(3 * bodies_.size(), 3 * bodies_.size());
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
        I.setIdentity();
        qprev_t = q_t;
        q_t += h * qdot_t;
        computeTranslationalForceAndHessian(q_t, qprev_t, F, H, h);
        Eigen::VectorXd R = q_t - qprev_t - (h * qdot_t) - (h * h * Minv * F);
        auto iters = params_.NewtonMaxIters;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        while (R.norm() > params_.NewtonTolerance && iters) {
            A = I + ((h * h) * Minv * H);
            b = -R;
            solver.compute(A);
            if (solver.info() != Eigen::Success) {
                assert(false);
            }
            Eigen::VectorXd deltaq = solver.solve(b);
            if (solver.info() != Eigen::Success) {
                assert(false);
            }
            q_t += deltaq;
            iters--;
            computeTranslationalForceAndHessian(q_t, qprev_t, F, H, params_.timeStep);
            R = q_t - qprev_t - (h * qdot_t) - (h * h * Minv * F);
        }
        qdot_t += h * Minv * F;
        q.segment(0, bodies_.size() * 3) = q_t;
        qdot.segment(0, bodies_.size() * 3) = qdot_t;
    }

    {
        // rotational forces
        Eigen::VectorXd q_r = q.segment(bodies_.size() * 3, bodies_.size() * 3);
        Eigen::VectorXd qdot_r = qdot.segment(bodies_.size() * 3, bodies_.size() * 3);
        Eigen::VectorXd qprev_r = q_r;

        Eigen::VectorXd F(3 * bodies_.size());
        F.setZero();
        Eigen::SparseMatrix<double> H(3 * bodies_.size(), 3 * bodies_.size());

        // explicit theta_{i+1}
        for (int i = 0; i < bodies_.size(); i++) {
            q_r.segment<3>(i * 3) =
                VectorMath::thetaNext(
                    q_r.segment<3>(i * 3),
                    qdot_r.segment<3>(i * 3),
                    h
                );
        }

        computeRotationalForceAndHessian(q_r, qprev_r, F, H, params_.timeStep);

        auto iters = params_.NewtonMaxIters;
        Eigen::VectorXd R(3 * bodies_.size());
        Eigen::VectorXd delta(3 * bodies_.size());
        auto qdot_rp1 = qdot_r;

        compute_residualOmega(R, q_r, qdot_r, qdot_rp1, F);

        while (R.norm() > params_.NewtonTolerance && iters) {
            compute_deltaOmega(delta, R, q_r, qdot_rp1);

            qdot_rp1 += delta;

            iters--;
            compute_residualOmega(R, q_r, qdot_r, qdot_rp1, F);
        }

        qdot_r = qdot_rp1;
        q.segment(bodies_.size() * 3, bodies_.size() * 3) = q_r;
        qdot.segment(bodies_.size() * 3, bodies_.size() * 3) = qdot_r;
    }
}


void simulateOneStep()
{
    time_ += params_.timeStep;
    Eigen::VectorXd q(bodies_.size() * 6);
    Eigen::VectorXd qdot(bodies_.size() * 6);
    build_dofs(q, qdot);

    numericalIntegration(q, qdot);

    unbuild_dofs(q, qdot);


    // TODO: Gather DOFs, compute forces, integrate time, write DOFs back to rigid bodies

}

void callback()
{
    ImGui::SetNextWindowSize(ImVec2(500., 0.));
    ImGui::Begin("UI", nullptr);

    if (ImGui::Button("Recenter Camera", ImVec2(-1, 0)))
    {
        polyscope::view::resetCameraToHomeView();
    }

    if (ImGui::CollapsingHeader("Simulation Control", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Run/Pause Sim", ImVec2(-1, 0)))
        {
            running_ = !running_;
        }
        if (ImGui::Button("Reset Sim", ImVec2(-1, 0)))
        {
            running_ = false;
            initSimulation();
        }        
    }
    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputText("Filename", &sceneFile_);
        if (ImGui::Button("Load Scene", ImVec2(-1, 0)))
        {
            loadScene();
            initSimulation();
        }
    }
    if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputDouble("Timestep", &params_.timeStep);
        ImGui::InputDouble("Newton Tolerance", &params_.NewtonTolerance);
        ImGui::InputInt("Newton Max Iters", &params_.NewtonMaxIters);
    }
    if (ImGui::CollapsingHeader("Forces", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Gravity Enabled", &params_.gravityEnabled);
        ImGui::InputDouble("Gravity G", &params_.gravityG);
    }    

    ImGui::End();
}

int main(int argc, char **argv) 
{
  polyscope::view::setWindowSize(1600, 800);
  polyscope::options::buildGui = false;
  polyscope::options::openImGuiWindowForUserCallback = false;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  polyscope::options::autocenterStructures = false;
  polyscope::options::autoscaleStructures = false;
  polyscope::options::maxFPS = -1;

  sceneFile_ = "box.scn";

  initSimulation();

  polyscope::init();

  polyscope::state::userCallback = callback;

  while (!polyscope::render::engine->windowRequestsClose())
  {
      if (running_)
          simulateOneStep();
      updateRenderGeometry();
      auto * surf = polyscope::registerSurfaceMesh("Bodies", renderQ, renderF);
      surf->setTransparency(0.9);      

      polyscope::frameTick();
  }

  return 0;
}

