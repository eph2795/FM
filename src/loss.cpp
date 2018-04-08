#include <algorithm>

#include "loss.h"


inline double MSE::compute_loss(double prediction, double target) {
    return std::pow(prediction - target, 2);
}


double MSE::compute_loss(const Y& prediction, const Y& y) {
    double result = 0;
    for (size_t i = 0; i < prediction._targets.size(); i++) {
        // std::cout << "Prediction: " << one._targets[i] << ", Target: " << another._targets[i] << std::endl; 
        result += std::pow(prediction._targets[i] - y._targets[i], 2);
    }
    result /= prediction._targets.size();
    return result;
}


inline double MSE::compute_grad(double prediction, double target) {
    return 2 * (prediction - target);
}