#include <iostream>

#include <cmath>
#include <algorithm>

#include <utility>
#include <vector>

#include "data.h"
#include "model.h"


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


LinearWeights::LinearWeights(size_t features_number)
        : _features_number(features_number), _w(features_number) 
{}


inline void LinearWeights::update_weights(const LinearSparseWeights& update, double coef) {
    _w0 += coef * update._w0;
    for (std::pair<size_t, double> item: update._w._items) {
        _w[item.first] += coef * item.second;
    }
}


FMWeights::FMWeights(size_t features_number, size_t factors_size)
        : _features_number(features_number), _factors_size(factors_size)
        ,_w(_features_number), _v(_features_number, std::vector<double>(factors_size, std::sqrt(1.0 / factors_size)))
{}


inline void FMWeights::update_weights(const FMSparseWeights& update, double coef) {
    _w0 += coef * update._w0;
    for (std::pair<size_t, double> item: update._w._items) {
        _w[item.first] += coef * item.second;
    }
    for (std::pair<size_t, std::vector<double>> factors: update._v) {
        for (size_t factor_num = 0; factor_num < _factors_size; factor_num++) {
            _v[factors.first][factor_num] += coef * factors.second[factor_num];
        }
    }
}


Model::~Model() {}


LinearModel::LinearModel(size_t features_number, bool use_offset)
        : _use_offset(use_offset), _weights(features_number)
{}


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
    prediction._targets.resize(x._objects.size());
    for (size_t obj_idx = 0; obj_idx < x._objects.size(); obj_idx++) {
        prediction._targets[obj_idx] = predict(x._objects[obj_idx]);
    }
    return prediction;
}


void LinearModel::train(bool state) {
    if (state) {
        _grad = new LinearSparseWeights();
    } else {
        delete _grad;
    }
}


inline SparseWeights* LinearModel::compute_grad(const SparseVector& object) {
    _grad->_w0 = 1;
    _grad->_w = object;
    return _grad;
}


inline void LinearModel::update_weights(const SparseWeights* update, double coef) {
    const LinearSparseWeights* linear_update = dynamic_cast<const LinearSparseWeights*>(update);
    _weights.update_weights(*linear_update, coef);
    // _weights._w0 += coef * linear_update->_w0;
    // for (std::pair<size_t, double> feature: linear_update->_w._items) {
    //     _weights._w[feature.first] += coef * feature.second;
    // }
}


FMModel::FMModel(size_t features_number, size_t factors_size, bool use_offset)
        : _use_offset(use_offset), _weights(features_number, factors_size)
        , _precomputed_sp(factors_size)
{}


inline double FMModel::predict(const SparseVector& object) {
    double prediction = 0;
    if (_use_offset) {
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
            // std::cout << _weights._v[feature.first][factor_num] << " ";
        }
        // std::cout << std::endl << std::endl;
        _precomputed_sp[factor_num] = first;
        prediction += 0.5 * (std::pow(first, 2) - second);
    }
    return prediction;
}



Y FMModel::predict(const X& x) {
    Y prediction;
    prediction._targets.resize(x._objects.size());
    for (size_t obj_idx = 0; obj_idx < x._objects.size(); obj_idx++) {
        prediction._targets[obj_idx] = predict(x._objects[obj_idx]);
    }
    return prediction;
}


void FMModel::train(bool state) {
    if (state) {
        _grad = new FMSparseWeights();
    } else {
        delete _grad;
    }
}


inline SparseWeights* FMModel::compute_grad(const SparseVector& object) {
    _grad->_w0 = 1;
    _grad->_w = object;
    _grad->_v = std::unordered_map<size_t, std::vector<double>>();
    for (std::pair<size_t, double> feature: object._items) {
        _grad->_v[feature.first] = std::vector<double>(_weights._factors_size);
        for (size_t factor_num = 0; factor_num < _weights._factors_size; factor_num++) {
            double term = _precomputed_sp[factor_num];
            // for (std::pair<size_t, double> feature_j: object._items) {
            //     term += _weights._v[feature_j.first][factor_num] * feature_j.second;
            // }
            _grad->_v[feature.first][factor_num] = feature.second * (term - _weights._v[feature.first][factor_num] * feature.second);
        }
    }
    return _grad;
}


inline void FMModel::update_weights(const SparseWeights* update, double coef) {
    const FMSparseWeights* fm_update = dynamic_cast<const FMSparseWeights*>(update);
    _weights.update_weights(*fm_update, coef);
    // _weights._w0 += coef * linear_update->_w0;
    // for (std::pair<size_t, double> feature: linear_update->_w._items) {
    //     _weights._w[feature.first] += coef * feature.second;
    // }
}
