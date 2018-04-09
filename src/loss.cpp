#include <cmath>

#include <algorithm>

#include "loss.h"


Loss::~Loss() {}


inline double MSE::compute_loss(double prediction, double target) {
    return std::pow(prediction - target, 2);
}


double MSE::compute_loss(const Y& prediction, const Y& y) {
    double result = 0;
    for (size_t i = 0; i < prediction._targets.size(); i++) {
        // std::cout << "Prediction: " << one._targets[i] << ", Target: " << another._targets[i] << std::endl; 
        result += compute_loss(prediction._targets[i], y._targets[i]);
    }
    result /= prediction._targets.size();
    return result;
}


inline double MSE::compute_grad(double prediction, double target) {
    return 2 * (prediction - target);
}


// Поменять target на +- 1
inline double Logistic::compute_loss(double prediction, double target) {
    return 1 / std::log(2) * std::log(1 + std::exp(-(2 * target - 1) * prediction));
}


double Logistic::compute_loss(const Y& prediction, const Y& y) {
    double result = 0;
    for (size_t i = 0; i < prediction._targets.size(); i++) {
        // std::cout << "Prediction: " << one._targets[i] << ", Target: " << another._targets[i] << std::endl; 
        result += compute_loss(prediction._targets[i], y._targets[i]);
    }
    result /= prediction._targets.size();
    return result;
}


inline double Logistic::compute_grad(double prediction, double target) {
    return 1 / std::log(2) * (-(2 * target - 1) * std::exp(-(2 * target - 1) * prediction)) / (1 + std::exp(-(2 * target - 1) * prediction));
}