#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>

class FilterConfiguration {
public:
    Eigen::Matrix3d V; // Process noise covariance
    double W_range;    // Measurement noise variance (range)
    double W_bearing;  // Measurement noise variance (bearing)
    Eigen::Vector3d x0;
    Eigen::Matrix3d Sigma0;

    FilterConfiguration() {
        // Process and measurement noise covariance matrices
        V = Eigen::Matrix3d::Identity();
        V.diagonal() << 0.1 * 0.1, 0.1 * 0.1, 0.05 * 0.05;

        // Measurement noise variance
        W_range = 0.5 * 0.5;
        W_bearing = std::pow(M_PI * 0.5 / 180.0, 2);

        // Initial conditions for the filter
        x0 << 2.0, 3.0, M_PI / 4;
        Sigma0 = Eigen::Matrix3d::Identity();
        Sigma0.diagonal() << 1.0, 1.0, 0.5;
    }
};

class Map {
public:
    std::vector<Eigen::Vector2d> landmarks;

    Map() {
        int n = 10;
        double range_min = -20;
        double range_max = 20;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double x = range_min + (range_max - range_min) * i / (n - 1);
                double y = range_min + (range_max - range_min) * j / (n - 1);
                landmarks.emplace_back(x, y);
            }
        }
    }
};

class RobotEstimator {
public:
    RobotEstimator(const FilterConfiguration& config, const Map& map)
        : _config(config), _map(map), _t(0) {}

    void start() {
        _t = 0;
        _set_estimate_to_initial_conditions();
    }

    void set_control_input(const Eigen::Vector2d& u) {
        _u = u;
    }

    void predict_to(double time) {
        double dt = time - _t;
        _t = time;
        _predict_over_dt(dt);
    }

    std::pair<Eigen::Vector3d, Eigen::Matrix3d> estimate() const {
        return {_x_est, _Sigma_est}; 
    }
    
    Eigen::Matrix3d Sigma_est() const { return _Sigma_est; }

    void copy_prediction_to_estimate() {
        _x_est = _x_pred;
        _Sigma_est = _Sigma_pred;
    }

    void update_from_landmark_range_observations(const Eigen::VectorXd& y_range) {
        Eigen::VectorXd y_pred(_map.landmarks.size());
        Eigen::MatrixXd C(_map.landmarks.size(), 3);

        for (size_t i = 0; i < _map.landmarks.size(); ++i) {
            Eigen::Vector2d lm = _map.landmarks[i];
            double dx_pred = lm[0] - _x_pred[0];
            double dy_pred = lm[1] - _x_pred[1];
            double range_pred = std::sqrt(dx_pred * dx_pred + dy_pred * dy_pred);
            y_pred[i] = range_pred;

            Eigen::RowVector3d C_range;
            C_range << -dx_pred / range_pred, -dy_pred / range_pred, 0;
            C.row(i) = C_range;
        }

        Eigen::VectorXd nu = y_range - y_pred;
        Eigen::MatrixXd W_landmarks = _config.W_range * Eigen::MatrixXd::Identity(_map.landmarks.size(), _map.landmarks.size());
        _do_kf_update(nu, C, W_landmarks);

        _x_est[2] = std::atan2(std::sin(_x_est[2]), std::cos(_x_est[2]));
    }

private:
    FilterConfiguration _config;
    Map _map;
    double _t;
    Eigen::Vector3d _x_est, _x_pred;
    Eigen::Matrix3d _Sigma_est, _Sigma_pred;
    Eigen::Vector2d _u;

    void _set_estimate_to_initial_conditions() {
        _x_est = _config.x0;
        _Sigma_est = _config.Sigma0;
    }

    void _predict_over_dt(double dt) {
        double v_c = _u[0];
        double omega_c = _u[1];

        _x_pred = _x_est + Eigen::Vector3d(
            v_c * std::cos(_x_est[2]) * dt,
            v_c * std::sin(_x_est[2]) * dt,
            omega_c * dt
        );
        _x_pred[2] = std::atan2(std::sin(_x_pred[2]), std::cos(_x_pred[2]));

        Eigen::Matrix3d A;
        A << 1, 0, -v_c * std::sin(_x_est[2]) * dt,
             0, 1,  v_c * std::cos(_x_est[2]) * dt,
             0, 0, 1;

        _kf_predict_covariance(A, _config.V * dt);
    }

    void _kf_predict_covariance(const Eigen::Matrix3d& A, const Eigen::Matrix3d& V) {
        _Sigma_pred = A * _Sigma_est * A.transpose() + V;
    }

    void _do_kf_update(const Eigen::VectorXd& nu, const Eigen::MatrixXd& C, const Eigen::MatrixXd& W) {
        Eigen::MatrixXd SigmaXZ = _Sigma_pred * C.transpose();
        Eigen::MatrixXd SigmaZZ = C * SigmaXZ + W;
        Eigen::MatrixXd K = SigmaXZ * SigmaZZ.inverse();

        _x_est = _x_pred + K * nu;
        _Sigma_est = (Eigen::Matrix3d::Identity() - K * C) * _Sigma_pred;
    }
};
