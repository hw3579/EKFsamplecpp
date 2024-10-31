#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <random>
#include "tqdm/tqdm.h"
#include "robot_localization_system.hpp" // 包含FilterConfiguration, Map, RobotEstimator的头文件

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

        // 使用初始状态和协方差生成真实状态
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

int main() {
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

    std::tie(x_est, Sigma_est) = estimator.estimate();
    u = controller.next_control_input(x_est, Sigma_est);

    std::vector<Eigen::Vector3d> x_true_history;
    std::vector<Eigen::Vector3d> x_est_history;
    std::vector<Eigen::Vector3d> Sigma_est_history;

    // 创建进度条对象
    std::vector<int> steps(sim_config.time_steps);
    std::iota(steps.begin(), steps.end(), 0);

    for (int step : tqdm::tqdm(steps)) {

        // 设置控制输入并在模拟器中传播步骤
        simulator.set_control_input(u);
        double simulation_time = simulator.step();

        // 使用相同的控制输入预测Kalman滤波器至相同时间
        estimator.set_control_input(u);
        estimator.predict_to(simulation_time);

        // 获取地标观测
        Eigen::VectorXd y = simulator.landmark_range_observations();

        // 使用最新的观测更新滤波器
        estimator.update_from_landmark_range_observations(y);

        // 获取当前状态估计
        std::tie(x_est, Sigma_est) = estimator.estimate();
        Sigma_est = estimator.Sigma_est();

        // 确定下一步控制指令
        u = controller.next_control_input(x_est, Sigma_est);

        // 存储数据以便绘图
        x_true_history.push_back(simulator.x_true());
        x_est_history.push_back(x_est);
        Sigma_est_history.push_back(Sigma_est.diagonal());
    }

    // 将数据从 vector 转换为 Eigen::Matrix 以便于进一步处理或分析
    Eigen::MatrixXd x_true_history_matrix(x_true_history.size(), 3);
    Eigen::MatrixXd x_est_history_matrix(x_est_history.size(), 3);
    Eigen::MatrixXd Sigma_est_history_matrix(Sigma_est_history.size(), 3);

    for (size_t i = 0; i < x_true_history.size(); ++i) {
        x_true_history_matrix.row(i) = x_true_history[i];
        x_est_history_matrix.row(i) = x_est_history[i];
        Sigma_est_history_matrix.row(i) = Sigma_est_history[i];
    }

    std::cout << "Simulation complete!" << std::endl;
    return 0;
}