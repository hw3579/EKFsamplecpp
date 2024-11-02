#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <random>
#include "matplotlibcpp.h"
#include "robot_localization_system.hpp" // ����FilterConfiguration, Map, RobotEstimator��ͷ�ļ�
#include <omp.h>
#include <chrono>
#include <thread>

namespace plt = matplotlibcpp;

class SimulatorConfiguration {
public:
    double dt;
    double total_time;
    int time_steps;
    double v_c;
    double omega_c;

    SimulatorConfiguration() {
        dt = 0.1;
        total_time = 1000.0;
        time_steps = static_cast<int>(total_time / dt);
        v_c = 1.0;
        omega_c = 0.1;
    }
};

class Controller {
public:
    Controller(const SimulatorConfiguration& config) : _config(config) {}

    Eigen::Vector2d next_control_input(const Eigen::Vector3d& x_est, const Eigen::Matrix3d& Sigma_est) {
        return Eigen::Vector2d(_config.v_c, _config.omega_c);
    }

private:
    SimulatorConfiguration _config;
};

class Simulator {
public:
    Simulator(const SimulatorConfiguration& sim_config, const FilterConfiguration& filter_config, const Map& map)
        : _config(sim_config), _filter_config(filter_config), _map(map), _time(0.0) {}

    void start() {
        _time = 0;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 1);

        // ʹ�ó�ʼ״̬��Э����������ʵ״̬
        _x_true = _filter_config.x0;
        Eigen::LLT<Eigen::Matrix3d> llt(_filter_config.Sigma0);
        _x_true += llt.matrixL() * Eigen::Vector3d(d(gen), d(gen), d(gen));
        _u = Eigen::Vector2d(0, 0);
    }

    void set_control_input(const Eigen::Vector2d& u) {
        _u = u;
    }

    double step() {
        double dt = _config.dt;
        double v_c = _u[0];
        double omega_c = _u[1];

        Eigen::Vector3d v = sample_process_noise(_filter_config.V * dt);

        _x_true += Eigen::Vector3d(
            v_c * std::cos(_x_true[2]) * dt,
            v_c * std::sin(_x_true[2]) * dt,
            omega_c * dt
        ) + v;

        _x_true[2] = std::atan2(std::sin(_x_true[2]), std::cos(_x_true[2]));
        _time += dt;
        return _time;
    }

    Eigen::VectorXd landmark_range_observations() {
        Eigen::VectorXd y(_map.landmarks.size());
        double W = _filter_config.W_range;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> noise(0, std::sqrt(W));

        for (size_t i = 0; i < _map.landmarks.size(); ++i) {
            const Eigen::Vector2d& lm = _map.landmarks[i];
            double dx = lm[0] - _x_true[0];
            double dy = lm[1] - _x_true[1];
            double range_true = std::sqrt(dx * dx + dy * dy);
            double range_meas = range_true + noise(gen);
            y[i] = range_meas;
        }
        return y;
    }

    Eigen::Vector3d x_true() const {
        return _x_true;
    }

private:
    Eigen::Vector3d sample_process_noise(const Eigen::Matrix3d& cov) {
        Eigen::Vector3d noise;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 1);

        Eigen::LLT<Eigen::Matrix3d> llt(cov);
        noise = llt.matrixL() * Eigen::Vector3d(d(gen), d(gen), d(gen));
        return noise;
    }

    SimulatorConfiguration _config;
    FilterConfiguration _filter_config;
    Map _map;
    double _time;
    Eigen::Vector3d _x_true;
    Eigen::Vector2d _u;
};

void plot_path(const std::vector<std::vector<double>>& x_true_history,
    const std::vector<std::vector<double>>& x_est_history,
    const std::vector<std::vector<double>>& landmarks) {

    
    plt::figure();

    // ������ʵ·��
    std::vector<double> x_true_x, x_true_y;
    for (const auto& point : x_true_history) {
        x_true_x.push_back(point[0]);
        x_true_y.push_back(point[1]);
    }
    plt::plot(x_true_x, x_true_y, { {"label", "True Path"} });

    // ���ƹ���·��
    std::vector<double> x_est_x, x_est_y;
    for (const auto& point : x_est_history) {
        x_est_x.push_back(point[0]);
        x_est_y.push_back(point[1]);
    }
    plt::plot(x_est_x, x_est_y, { {"label", "Estimated Path"} });

    // ���Ƶر��
    std::vector<double> landmarks_x, landmarks_y;
    for (const auto& landmark : landmarks) {
        landmarks_x.push_back(landmark[0]);
        landmarks_y.push_back(landmark[1]);
    }
    plt::scatter(landmarks_x, landmarks_y, 10.0, { {"color", "red"}, {"label", "Landmarks"}, {"marker", "x"} });

    // ����ͼ������ǩ�����⡢�������
    plt::legend();
    plt::xlabel("X position [m]");
    plt::ylabel("Y position [m]");
    plt::title("Unicycle Robot Localization using EKF");
    plt::axis("equal");
    plt::grid(true);

    // ��ʾͼ��
    plt::show();
}

double wrap_angle(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

void plot_estimation_error(const Eigen::MatrixXd& x_est_history_matrix, const Eigen::MatrixXd& x_true_history_matrix, const Eigen::MatrixXd& Sigma_est_history_matrix) {
    std::vector<std::string> state_name = { "x", "y", "��" };
    std::vector<std::string> state_name2 = { "x", "y", "theta" };

    Eigen::MatrixXd estimation_error = x_est_history_matrix - x_true_history_matrix;
    for (int i = 0; i < estimation_error.rows(); ++i) {
        estimation_error(i, 2) = wrap_angle(estimation_error(i, 2));
    }

    for (int s = 0; s < 3; ++s) {

        plt::figure();

        Eigen::VectorXd two_sigma = 2 * Sigma_est_history_matrix.col(s).array().sqrt();

        std::vector<double> error(estimation_error.col(s).data(), estimation_error.col(s).data() + estimation_error.rows());
        std::vector<double> two_sigma_vec(two_sigma.data(), two_sigma.data() + two_sigma.size());
        std::vector<double> neg_two_sigma_vec = two_sigma_vec;
        for (auto& val : neg_two_sigma_vec) {
            val = -val;
        }

        plt::plot(error, { {"label", "error"} });
        plt::plot(two_sigma_vec, { {"linestyle", "dashed"}, {"color", "red"}, {"alpha", "0.7"}, {"label", "95% confidence"} });
        plt::plot(neg_two_sigma_vec, { {"linestyle", "dashed"}, {"color", "red"}, {"alpha", "0.7"} });

        if (s == 0 || s == 1) {
            plt::plot({ 0, static_cast<double>(two_sigma_vec.size()) }, { 0.1, 0.1 }, { {"color", "tab:orange"}, {"linestyle", ":"}, {"label", "10 cm"} });
            plt::plot({ 0, static_cast<double>(two_sigma_vec.size()) }, { -0.1, -0.1 }, { {"color", "tab:orange"}, {"linestyle", ":"} });
        }

        //plt::legend();
        //plt::title(state_name[s]);
        //plt::xlabel("localization error (m)");
        //plt::ylabel("time step (0.1s)");
        plt::show();

    }
}



int main() {
    // ��¼��ʼʱ��
    auto start = std::chrono::high_resolution_clock::now();

    unsigned int num_threads = std::thread::hardware_concurrency();
    Eigen::setNbThreads(num_threads - 1);



    SimulatorConfiguration sim_config;
    FilterConfiguration filter_config;
    Map map;

    Controller controller(sim_config);
    Simulator simulator(sim_config, filter_config, map);
    simulator.start();

    RobotEstimator estimator(filter_config, map);
    estimator.start();

    Eigen::Vector3d x_est;
    Eigen::Matrix3d Sigma_est;
    Eigen::Vector2d u;

    x_est = estimator.estimate();
    u = controller.next_control_input(x_est, Sigma_est);

    std::vector<Eigen::Vector3d> x_true_history;
    std::vector<Eigen::Vector3d> x_est_history;
    std::vector<Eigen::Vector3d> Sigma_est_history;

    #pragma omp parallel for
    for (int step = 0; step < sim_config.time_steps; ++step) {
        //bar.set_progress(static_cast<size_t>(100.0 * step / sim_config.time_steps));

        // ���ÿ������벢��ģ�����д�������
        simulator.set_control_input(u);
        double simulation_time = simulator.step();

        // ʹ����ͬ�Ŀ�������Ԥ��Kalman�˲�������ͬʱ��
        estimator.set_control_input(u);
        estimator.predict_to(simulation_time);

        // ��ȡ�ر�۲�
        Eigen::VectorXd y = simulator.landmark_range_observations();

        // ʹ�����µĹ۲�����˲���
        estimator.update_from_landmark_range_observations(y);

        // ��ȡ��ǰ״̬����
        x_est = estimator.estimate();
        Sigma_est = estimator.getSigmaEst();

        // ȷ����һ������ָ��
        u = controller.next_control_input(x_est, Sigma_est);

        // �洢�����Ա��ͼ
        x_true_history.push_back(simulator.x_true());
        x_est_history.push_back(x_est);
        Sigma_est_history.push_back(Sigma_est.diagonal());

        //��ʾ������
        std::cout << "Simulation progress: " << step << "/" << sim_config.time_steps << std::endl;
    }

    // ��¼����ʱ��
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // �������ʱ��
    std::cout << "����ʱ��: " << duration.count() << " ��" << std::endl;

    // �����ݴ� vector ת��Ϊ Eigen::Matrix �Ա��ڽ�һ����������
    Eigen::MatrixXd x_true_history_matrix(x_true_history.size(), 3);
    Eigen::MatrixXd x_est_history_matrix(x_est_history.size(), 3);
    Eigen::MatrixXd Sigma_est_history_matrix(Sigma_est_history.size(), 3);

    for (size_t i = 0; i < x_true_history.size(); ++i) {
        x_true_history_matrix.row(i) = x_true_history[i];
        x_est_history_matrix.row(i) = x_est_history[i];
        Sigma_est_history_matrix.row(i) = Sigma_est_history[i];
    }

    std::cout << "Simulation complete!" << std::endl;

    // ʹ��matplotlib���л�ͼ
    std::vector<std::vector<double>> x_true_history_vec(x_true_history.size(), std::vector<double>(3));
    std::vector<std::vector<double>> x_est_history_vec(x_est_history.size(), std::vector<double>(3));
    std::vector<std::vector<double>> landmarks_vec(map.landmarks.size(), std::vector<double>(2));

    for (size_t i = 0; i < x_true_history.size(); ++i) {
        x_true_history_vec[i][0] = x_true_history[i][0];
        x_true_history_vec[i][1] = x_true_history[i][1];
        x_true_history_vec[i][2] = x_true_history[i][2];
    }

    for (size_t i = 0; i < x_est_history.size(); ++i) {
        x_est_history_vec[i][0] = x_est_history[i][0];
        x_est_history_vec[i][1] = x_est_history[i][1];
        x_est_history_vec[i][2] = x_est_history[i][2];
    }

    for (size_t i = 0; i < map.landmarks.size(); ++i) {
        landmarks_vec[i][0] = map.landmarks[i][0];
        landmarks_vec[i][1] = map.landmarks[i][1];
    }

    plot_path(x_true_history_vec, x_est_history_vec, landmarks_vec);

    for (size_t i = 0; i < x_true_history.size(); ++i) {
        x_true_history_matrix.row(i) = x_true_history[i];
        x_est_history_matrix.row(i) = x_est_history[i];
        Sigma_est_history_matrix.row(i) = Sigma_est_history[i];
    }

    plot_estimation_error(x_est_history_matrix, x_true_history_matrix, Sigma_est_history_matrix);

    return 0;
}