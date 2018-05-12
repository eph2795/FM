#include <iostream>
#include <string.h>
#include <cmath>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "data.h"
#include "model.h"
#include "regularizer.h"


X _to_csr(const X& x) {
    X x_csr;
    x_csr._features_number = x._features_number;
    x_csr._objects_number = x._objects_number;
    x_csr._data_type = "csr";

    x_csr._objects.resize(x_csr._objects_number);
    size_t object_idx;
    double feature_value;

    for (size_t feature_idx = 0; feature_idx < x._features_number; feature_idx++) {
        for (const auto& item: x._objects[feature_idx]._items) {
            object_idx = item.first;
            feature_value = item.second;
            x_csr._objects[object_idx]._items.push_back(std::pair<size_t, double>(feature_idx, feature_value));
        }
    } 
    return x_csr;
}


SparseWeights::~SparseWeights() {}

// LinearSparseWeights& LinearSparseWeights::operator+(const LinearSparseWeights& another) {
//     _w0 += another._w0;
//     for (std::pair<size_t, double> item: another._w) {
//         if _w[item.first] += item.second; 
//     }
//     return *this;
// }


// SparseWeights* LinearWeights::compute_grad(const SparseVector& object) {
//     SparseWeights* grad = new SparseWeights;
//     grad->_w0 = 1;
//     grad->_w = object;
//     return grad;
// }


Weights::~Weights() {}


LinearWeights::LinearWeights(size_t features_number, Regularizer* regularizer)
        : _features_number(features_number), _w0(0), _w(features_number, 0), _regularizer(regularizer)
{}


LinearWeights::~LinearWeights() {
    delete _regularizer;
}


inline void LinearWeights::update_weights(const SparseWeights* update, double coef) {
    const LinearSparseWeights* linear_update = dynamic_cast<const LinearSparseWeights*>(update);
    _w0 += coef * (linear_update->_w0 + _regularizer->get_update(_w0));
    for (std::pair<size_t, double> item: linear_update->_w._items) {
        _w[item.first] += coef * (item.second + _regularizer->get_update(_w[item.first]));
    }
}


FMWeights::FMWeights(size_t features_number, size_t factors_size, Regularizer* regularizer)
        : _features_number(features_number), _factors_size(factors_size), _w0(0)
        ,_w(features_number, 0), _v(features_number, std::vector<double>(factors_size))
        ,_regularizer(regularizer)
{
     std::default_random_engine generator;
     std::normal_distribution<double> weights_sampler(0, 0.001);
     for (size_t i = 0; i < features_number; i++) {
        for (size_t j = 0; j < factors_size; j++) {
            _v[i][j] = weights_sampler(generator);
        }
     }

//     std::uniform_int_distribution<size_t> object_sampler(0, N);
//     std::uniform_int_distribution<size_t> gradient_sampler(0, _update_frequency);
}


FMWeights::~FMWeights() {
    delete _regularizer;
}


inline void FMWeights::update_weights(const SparseWeights* update, double coef) {
    const FMSparseWeights* fm_update = dynamic_cast<const FMSparseWeights*>(update);
    _w0 += coef * (fm_update->_w0 + _regularizer->get_update(_w0));
    for (std::pair<size_t, double> item: fm_update->_w._items) {
        _w[item.first] += coef * (item.second + _regularizer->get_update(_w[item.first]));
    }
    for (std::pair<size_t, std::vector<double>> factors: fm_update->_v) {
        for (size_t factor_num = 0; factor_num < _factors_size; factor_num++) {
            _v[factors.first][factor_num] += coef * (factors.second[factor_num] + _regularizer->get_update(_v[factors.first][factor_num]));
        }
    }
}


Model::~Model() {}


LinearModel::LinearModel(size_t features_number, bool use_offset, Regularizer* regularizer)
        : _state(false), _use_offset(use_offset), _weights(features_number, regularizer), _e(0)
{}


// LinearModel::~LinearModel() {
//     delete _grad;
// }


inline double LinearModel::predict(const SparseVector& object) {
    double prediction = 0;
    if (_use_offset) {
        prediction += _weights._w0;
    } 
    for (std::pair<size_t, double> feature: object._items) {
        prediction += _weights._w[feature.first] * feature.second;
    }
    return prediction;
}


Y LinearModel::predict(const X& x) {
    Y prediction;
    if (strcmp(x._data_type.c_str(), "csr") == 0) {
        prediction._targets.resize(x._objects_number);
        for (size_t obj_idx = 0; obj_idx < x._objects.size(); obj_idx++) {
            prediction._targets[obj_idx] = predict(x._objects[obj_idx]);
        }
    } else if (strcmp(x._data_type.c_str(), "csc") == 0) {
        prediction._targets.resize(x._objects_number, _weights._w0);
        size_t object_idx;
        double feature_value; 
        for (size_t feature_idx = 0; feature_idx < x._features_number; feature_idx++) {    
            for (const auto& item: x._objects[feature_idx]._items) {
                object_idx = item.first;
                feature_value = item.second;
                prediction._targets[object_idx] += _weights._w[feature_idx] * feature_value;
            }
        }
    } else {
        std::cout << "Wrong sparse matrix type! Terminated." << std::endl;
        throw;
    }
    return prediction;
}


void LinearModel::train(bool state, const std::string& method, size_t num_objects) {
    if (strcmp(method.c_str(), "sgd") == 0) {
        if (state and not _state) {
            _state = true;
            _grad = new LinearSparseWeights();
        } else if (not state and _state) {
            _state = false;
            delete _grad;
        }
    } else if (strcmp(method.c_str(), "als") == 0) {
        if (state and not _state) {
            _state = true;
            _e.resize(num_objects, 0);
            // _x_squares.resize(_weights._features_number, 0);
            // _numerators.resize(_weights._features_number, 0);
        }  else if (not state and _state) {
            _state = false;
            _e.resize(0);
            // _x_squares.resize(0);
            // _numerators.resize(0);
        } 
    }
}


Weights* LinearModel::get_weights() {
    return &_weights;
}


Model* LinearModel::clone() const {
    Regularizer* regularizer = _weights._regularizer->clone();
    LinearModel* clone = new LinearModel(_weights._features_number, _use_offset, regularizer);
    clone->_weights = _weights;
    return clone;
}


inline SparseWeights* LinearModel::compute_grad(const SparseVector& object, double coef) {
    _grad->_w0 = coef;
    _grad->_w = object;
    for (auto& feature: _grad->_w._items) {
        feature.second *= coef;
    }
    return _grad;
}


inline void LinearModel::update_weights(const SparseWeights* update, double coef) {
    const LinearSparseWeights* linear_update = dynamic_cast<const LinearSparseWeights*>(update);
    _weights.update_weights(linear_update, coef);
}


void LinearModel::init_als(Loss* loss, const X& x, const Y& y) {
    std::cout << _e.size() << " " << y._targets.size() << std::endl;
    for (size_t object_idx = 0; object_idx < x._objects_number; object_idx++) {
        _e[object_idx] = _weights._w0 - y._targets[object_idx];
    } 
    std::cout << x._objects.size() << std::endl;
    size_t object_idx;
    double feature_value;
    for (size_t feature_idx = 0; feature_idx < x._features_number; feature_idx++) {
        for (const auto& item: x._objects[feature_idx]._items) {
            object_idx = item.first;
            feature_value = item.second;
            _e[object_idx] += _weights._w[feature_idx] * feature_value; 
        }
    }
}


void LinearModel::als_step(Loss* loss, const X& x, const Y& y) {
    size_t M = x._features_number; // number of features
    size_t N = x._objects_number; // number of objects  

    double e_sum = 0;
    for (size_t object_idx = 0; object_idx < N; object_idx++) {
        e_sum += _e[object_idx];
    }

    double w_new = -(e_sum - N * _weights._w0) / (N + _weights._regularizer->get_C());
    for (size_t object_idx = 0; object_idx < N; object_idx++) {
        _e[object_idx] += w_new - _weights._w0;
    }
    _weights._w0 = w_new;


    std::vector<size_t> features_order(M);
    std::iota(features_order.begin(), features_order.end(), 0);
    std::random_shuffle(features_order.begin(), features_order.end());

    double numerator;
    double squares;
    size_t object_idx;
    double feature_value; 
    for (size_t feature_idx: features_order) {
        numerator = 0;
        squares = 0;
        for (const auto& item: x._objects[feature_idx]._items) {
            object_idx = item.first;
            feature_value = item.second;
            numerator += feature_value * (_e[object_idx] - _weights._w[feature_idx] * feature_value);
            squares += std::pow(feature_value, 2);
        }
        w_new = -numerator / (squares + _weights._regularizer->get_C());
        for (const auto& item: x._objects[feature_idx]._items) {
            object_idx = item.first;
            feature_value = item.second;
            _e[object_idx] += (w_new - _weights._w[feature_idx]) * feature_value;
            std::cout << _e[object_idx] << std::endl;
        }
        _weights._w[feature_idx] = w_new;
    }
}


void LinearModel::_linear_als_update() {
}


FMModel::FMModel(size_t features_number, size_t factors_size, bool use_offset, Regularizer* regularizer)
        : _state(false), _use_offset(use_offset), _weights(features_number, factors_size, regularizer)
        , _precomputed_sp(factors_size), _e(0), _q(0, std::vector<double>(0))
{}


// FMModel::~FMModel() {
//     delete _grad;
// }


inline double FMModel::predict(const SparseVector& object) {
    double prediction = 0;
    if (_use_offset) {
        // _weights._w0 += _regularizer(_weights._w0);
        prediction += _weights._w0;
    } 
    for (std::pair<size_t, double> feature: object._items) {
        prediction += _weights._w[feature.first] * feature.second;
    }
    for (size_t factor_num = 0; factor_num < _weights._factors_size; factor_num++) {
        double first = 0, second = 0, term;
        for (std::pair<size_t, double> feature: object._items) {
            term = _weights._v[feature.first][factor_num] * feature.second;
            first += term;
            second += std::pow(term, 2); 
        }
        _precomputed_sp[factor_num] = first;
        prediction += 0.5 * (std::pow(first, 2) - second);
    }
    return prediction;
}



Y FMModel::predict(const X& x) {
    X x_csr = _to_csr(x);
    Y prediction;
    prediction._targets.resize(x_csr._objects.size());
    for (size_t obj_idx = 0; obj_idx < x_csr._objects.size(); obj_idx++) {
        prediction._targets[obj_idx] = predict(x_csr._objects[obj_idx]);
    }
    return prediction;
}


void FMModel::train(bool state, const std::string& method, size_t num_objects) {
    if (state and not _state) {
        _state = true;
        _grad = new FMSparseWeights();
    } else if (not state and _state) {
        _state = false;
        delete _grad;
    }
}


Weights* FMModel::get_weights() {
    return &_weights;
}


inline SparseWeights* FMModel::compute_grad(const SparseVector& object, double coef) {
    _grad->_w0 = coef;
    _grad->_w = object;
    for (auto& feature: _grad->_w._items) {
        feature.second *= coef;
    }
    // _grad->_v = std::unordered_map<size_t, std::vector<double>>();
    _grad->_v.resize(0);
    for (auto feature: object._items) {
        // _grad->_v[feature.first] = std::vector<double>(_weights._factors_size);
        std::vector<double> new_grad(_weights._factors_size);
        for (size_t factor_num = 0; factor_num < _weights._factors_size; factor_num++) {
            double term = _precomputed_sp[factor_num];
            // _grad->_v[feature.first][factor_num] = feature.second * (term - _weights._v[feature.first][factor_num] * feature.second);
            new_grad[factor_num] = coef * feature.second * (term - _weights._v[feature.first][factor_num] * feature.second);
        }
        _grad->_v.push_back(std::pair<size_t, std::vector<double>>(feature.first, new_grad));
    }
    return _grad;
}


inline void FMModel::update_weights(const SparseWeights* update, double coef) {
    const FMSparseWeights* fm_update = dynamic_cast<const FMSparseWeights*>(update);
    _weights.update_weights(fm_update, coef);
}


void FMModel::init_als(Loss* loss, const X& x, const Y& y) {
    X x_csr = _to_csr(x);

    _e.resize(x._objects_number, 0);
    _q.resize(x._objects_number, std::vector<double>(_weights._factors_size, 0));

    for (size_t object_idx = 0; object_idx < x_csr._objects_number; object_idx++) {
        _e[object_idx] = predict(x_csr._objects[object_idx]) - y._targets[object_idx];
    } 
    
    size_t feature_idx;
    double feature_value;
    for (size_t factor_idx = 0; factor_idx < _weights._factors_size; factor_idx++) {
        for (size_t object_idx = 0; object_idx < x_csr._objects_number; object_idx++) {
            for (const auto& item: x_csr._objects[object_idx]._items) {
                feature_idx = item.first;
                feature_value = item.second;
                _q[object_idx][factor_idx] += _weights._v[feature_idx][factor_idx] * feature_value;
            }
        }
    }
}


void FMModel::als_step(Loss* loss, const X& x, const Y& y) {
    size_t M = x._features_number; // number of features
    size_t N = x._objects_number; // number of objects  

    double e_sum = 0;
    for (size_t object_idx = 0; object_idx < N; object_idx++) {
        e_sum += _e[object_idx];
    }

    double w_new = -(e_sum - N * _weights._w0) / (N + _weights._regularizer->get_C());
    for (size_t object_idx = 0; object_idx < N; object_idx++) {
        _e[object_idx] += w_new - _weights._w0;
    }
    _weights._w0 = w_new;


    std::vector<size_t> features_order(M);
    std::iota(features_order.begin(), features_order.end(), 0);
    std::random_shuffle(features_order.begin(), features_order.end());

    double numerator, squares, h, feature_value;
    size_t object_idx;
    for (size_t feature_idx: features_order) {
        numerator = 0;
        squares = 0;
        for (const auto& item: x._objects[feature_idx]._items) {
            object_idx = item.first;
            feature_value = item.second;
            numerator += feature_value * (_e[object_idx] - _weights._w[feature_idx] * feature_value);
            squares += std::pow(feature_value, 2);
        }
        w_new = -numerator / (squares + _weights._regularizer->get_C());
        for (const auto& item: x._objects[feature_idx]._items) {
            object_idx = item.first;
            feature_value = item.second;
            _e[object_idx] += (w_new - _weights._w[feature_idx]) * feature_value;
        }
        _weights._w[feature_idx] = w_new;
    }

    std::random_shuffle(features_order.begin(), features_order.end());
    for (size_t factor_idx = 0; factor_idx < _weights._factors_size; factor_idx++) {
        for (size_t feature_idx: features_order) {

            numerator = 0;
            squares = 0;
            for (const auto& item: x._objects[feature_idx]._items) {
                object_idx = item.first;
                feature_value = item.second;
                
                h = feature_value * (_q[object_idx][factor_idx] - _weights._v[feature_idx][factor_idx] * feature_value);
                numerator += h * (_e[object_idx] - _weights._v[feature_idx][factor_idx] * h);
                squares += std::pow(h, 2);
            }
            
            w_new = -numerator / (squares + _weights._regularizer->get_C());
            for (const auto& item: x._objects[feature_idx]._items) {
                object_idx = item.first;
                feature_value = item.second;
                
                h = feature_value * (_q[object_idx][factor_idx] - _weights._v[feature_idx][factor_idx] * feature_value);
                _e[object_idx] += (w_new - _weights._v[feature_idx][factor_idx]) * h;
                _q[object_idx][factor_idx] += (w_new - _weights._v[feature_idx][factor_idx]) * feature_value;
            }
            _weights._v[feature_idx][factor_idx] = w_new;
        }
    }
}


Model* FMModel::clone() const {
    Regularizer* regularizer = _weights._regularizer->clone();
    FMModel* clone = new FMModel(_weights._features_number, _weights._factors_size, _use_offset, regularizer);
    clone->_weights = _weights;
    return clone;  
}