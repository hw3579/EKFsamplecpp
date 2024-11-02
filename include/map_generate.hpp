
#include <vector>
#include <Eigen/Dense>



class Map {
public:
    std::vector<Eigen::Vector2d> landmarks;

    Map() {
        int n = 11;
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