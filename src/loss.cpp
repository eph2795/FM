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
        result += compute_loss(prediction._targets[i], y._targets[i]);
    }
    result /= prediction._targets.size();
    return result;
}


inline double MSE::compute_grad(double prediction, double target) {
    return 2 * (prediction - target);
}


inline double Logistic::compute_loss(double prediction, double target) {
    // target = 2 * target - 1;
    return std::log(1 + std::exp(-target * prediction));
}


double Logistic::compute_loss(const Y& prediction, const Y& y) {
    double result = 0;
    for (size_t i = 0; i < prediction._targets.size(); i++) { 
        result += compute_loss(prediction._targets[i], y._targets[i]);
    }
    result /= prediction._targets.size();
    return result;
}


inline double Logistic::compute_grad(double prediction, double target) {
    // target = 2 * target - 1;
    double exp = std::exp(-target * prediction);
    return (-target * exp) / (1 + exp);
}