#include <iostream>

#include <algorithm>

#include <vector>

#include "data.h"
#include "optimization.h"


template<typename T>
void print_vector(const std::vector<T>& v) {
    std::cout << "Vector:" << std::endl;
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl << std::endl;
}


double scalar_product(const Object& object, const std::vector<double>& w) {
    double result = 0;
    for (Feature feature: object._features) {
        result += w[feature.idx] * feature.value;
    }
    return result;
}

double MSE_grad(const Model& model, const X& x, const Y& y, size_t obj_idx) {
    double result = 0;
    Object object = x._objects[obj_idx];
    for (Feature feature: object._features) {
        result += feature.value * model._w[feature.idx];
    }
    result = 2 * (result - y._targets[obj_idx]);
    return result;
}


std::vector<double> model_grad(const Model& model, const X& x, size_t obj_idx) {
    std::vector<double> result(model._w.size(), 0);
    Object object = x._objects[obj_idx];
    for (Feature feature: object._features) {
        result[feature.idx] = feature.value;
    }
    return result;
}


Optimizer::Optimizer(size_t num_epochs, double learning_rate, size_t batch_size)
        : _num_epochs(num_epochs), _learning_rate(learning_rate), _batch_size(batch_size)
{}


void Optimizer::train(Model* model, const X& x, const Y& y) {
    size_t N = x._objects.size();
    std::vector<size_t> objects_order(N);
    std::iota(objects_order.begin(), objects_order.end(), 0);

    for (size_t epoch = 0; epoch < _num_epochs; epoch++) {
        std::random_shuffle(objects_order.begin(), objects_order.end());

        size_t batch_N = N / _batch_size + (N % _batch_size != 0);
        // std::cout << "Data size: " << N << ", number of batches: " << batch_N << std::endl;
        for (size_t batch_i = 0; batch_i < batch_N; batch_i++) {
            std::vector<size_t> obj_idxes(objects_order.begin() + batch_i * _batch_size, 
                                          objects_order.begin() + std::min((batch_i + 1) * _batch_size, N));
            
            std::vector<double> update(model->_w.size(), 0);
            for (size_t obj_idx: obj_idxes) {
                double coef = -_learning_rate * MSE_grad(*model, x, y, obj_idx);
                // std::cout << "Coef for batch " << obj_idx << ": " << coef << std::endl;
                std::vector<double> grad = model_grad(*model, x, obj_idx);
                for (size_t k = 0; k < grad.size(); k++) {
                    update[k] += coef * grad[k];
                } 
            }

            // print_vector(update);
            for (size_t k = 0; k < update.size(); k++) {
                model->_w[k] += update[k] / _batch_size;
            }
            // print_vector(model->_w);
        }
        // print_vector(model->_w);
    }
}


Y Model::predict(const Model& model, const X& x) {
    Y y;
    for (const Object& object: x._objects) {
        y._targets.push_back(scalar_product(object, model._w));
    }
    return y;
}


double MSE(const Y& one, const Y& another) {
    double result = 0;
    for (size_t i = 0; i < one._targets.size(); i++) {
        std::cout << "Prediction: " << one._targets[i] << ", Target: " << another._targets[i] << std::endl; 
        result += std::pow(one._targets[i] - another._targets[i], 2);
    }
    result /= one._targets.size();
    return result;
}