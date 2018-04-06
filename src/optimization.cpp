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


void print_object(const Object& object) {
    for (Feature feature: object._features) {
        std::cout << "Col: " << feature.idx << ", " << "Val: " << feature.value << "\t";
    }
    std::cout << std::endl;
}


Object update(const Object& one, const Object& another, double coef) {
    Object result;
    size_t one_size = one._features.size();
    size_t another_size = another._features.size();
    size_t one_idx = 0;
    size_t another_idx = 0;
    while ((one_idx < one_size) || (another_idx < another_size)) {
        Feature feature;
        if ((one_idx == one_size) || (one._features[one_idx].idx > another._features[another_idx].idx)) {
            feature = another._features[another_idx];
            feature.value *= coef;
            another_idx += 1;
        } else if ((another_idx == another_size) || (one._features[one_idx].idx < another._features[another_idx].idx)) {
            feature = one._features[one_idx]; 
            one_idx += 1;
        } else {
            feature.idx = one._features[one_idx].idx;
            feature.value = one._features[one_idx].value + coef * another._features[another_idx].value;
            one_idx += 1;
            another_idx += 1;
        }
        result._features.push_back(feature);
    }
    return result;
}


void update(std::vector<double>* w, const Object& object, double coef) {
    for (Feature feature: object._features) {
        (*w)[feature.idx] += coef * feature.value;
    }
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


Object model_grad(const Model& model, const X& x, size_t obj_idx) {
    Object result(x._objects[obj_idx]);
    return result;
}


Optimizer::Optimizer(size_t num_epochs, double learning_rate)
        : _num_epochs(num_epochs), _learning_rate(learning_rate)
{}


void Optimizer::train(Model* model, const X& x, const Y& y) {
    size_t N = x._objects.size();
    // size_t f = model->_w.size();

    std::vector<size_t> objects_order(N);
    std::iota(objects_order.begin(), objects_order.end(), 0);

    // Object w_update;
    Object m_grad;
    for (size_t epoch = 0; epoch < _num_epochs; epoch++) {
        std::random_shuffle(objects_order.begin(), objects_order.end());

        // size_t batch_N = N / _batch_size + (N % _batch_size != 0);
        // std::cout << "Data size: " << N << ", number of batches: " << batch_N << std::endl;
        // size_t i = 0;
        for (size_t obj_idx: objects_order) {
            double coef = -_learning_rate * MSE_grad(*model, x, y, obj_idx);
            m_grad = model_grad(*model, x, obj_idx);
            update(&(model->_w),  m_grad, coef);
            // std::cout << "Obj idx: " << obj_idx << std::endl;
            // print_object(m_grad);
            // w_update = update(w_update, m_grad, coef);
            // std::cout << "Obj coef: " << coef << std::endl;
            
            // print_object(w_update);
            // break;
            // if (((i + 1) % _batch_size == 0) || (i == N - 1)) {
            //     // print_object(w_update);
            //     update(&(model->_w),  w_update, 1.0 / _batch_size);
            //     w_update._features.resize(0);      
            //     // print_vector(model->_w);
            //     // std::cout << "\n\n\n\n\n";
            //     // break;
            // }
            // // print_vector(model->_w);
            // i += 1;
        }
        // break;
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
        // std::cout << "Prediction: " << one._targets[i] << ", Target: " << another._targets[i] << std::endl; 
        result += std::pow(one._targets[i] - another._targets[i], 2);
    }
    result /= one._targets.size();
    return result;
}