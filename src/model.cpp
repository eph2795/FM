#include <iostream>

#include <cmath>
#include <algorithm>

#include <utility>
#include <vector>

#include "data.h"
#include "model.h"
#include "regularizer.h"


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


LinearWeights::LinearWeights(size_t features_number, Regularizer* regularizer)
        : _features_number(features_number), _w(features_number), _regularizer(regularizer)
{}


LinearWeights::~LinearWeights() {
    delete _regularizer;
}


inline void LinearWeights::update_weights(const LinearSparseWeights& update, double coef) {
    _w0 += coef * (update._w0 + _regularizer->get_update(_w0));
    for (std::pair<size_t, double> item: update._w._items) {
        _w[item.first] += coef * (item.second + _regularizer->get_update(_w[item.first]));
    }
}


FMWeights::FMWeights(size_t features_number, size_t factors_size, Regularizer* regularizer)
        : _features_number(features_number), _factors_size(factors_size)
        ,_w(_features_number), _v(_features_number, std::vector<double>(factors_size, std::sqrt(1.0 / factors_size)))
        ,_regularizer(regularizer)
{}


FMWeights::~FMWeights() {
    delete _regularizer;
}


inline void FMWeights::update_weights(const FMSparseWeights& update, double coef) {
    _w0 += coef * (update._w0 + _regularizer->get_update(_w0));
    for (std::pair<size_t, double> item: update._w._items) {
        _w[item.first] += coef * (item.second + _regularizer->get_update(_w[item.first]));
    }
    for (std::pair<size_t, std::vector<double>> factors: update._v) {
        for (size_t factor_num = 0; factor_num < _factors_size; factor_num++) {
            _v[factors.first][factor_num] += coef * (factors.second[factor_num] + _regularizer->get_update(_v[factors.first][factor_num]));
        }
    }
}


Model::~Model() {}


LinearModel::LinearModel(size_t features_number, bool use_offset, Regularizer* regularizer)
        : _state(false), _use_offset(use_offset), _weights(features_number, regularizer)
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
    prediction._targets.resize(x._objects.size());
    for (size_t obj_idx = 0; obj_idx < x._objects.size(); obj_idx++) {
        prediction._targets[obj_idx] = predict(x._objects[obj_idx]);
    }
    return prediction;
}


void LinearModel::train(bool state) {
    if (state and not _state) {
        _state = true;
        _grad = new LinearSparseWeights();
    } else if (not state and _state) {
        _state = false;
        delete _grad;
    }
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
    _weights.update_weights(*linear_update, coef);
}


FMModel::FMModel(size_t features_number, size_t factors_size, bool use_offset, Regularizer* regularizer)
        : _state(false), _use_offset(use_offset), _weights(features_number, factors_size, regularizer)
        , _precomputed_sp(factors_size)
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
    Y prediction;
    prediction._targets.resize(x._objects.size());
    for (size_t obj_idx = 0; obj_idx < x._objects.size(); obj_idx++) {
        prediction._targets[obj_idx] = predict(x._objects[obj_idx]);
    }
    return prediction;
}


void FMModel::train(bool state) {
    if (state and not _state) {
        _state = true;
        _grad = new FMSparseWeights();
    } else if (not state and _state) {
        _state = false;
        delete _grad;
    }
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
    _weights.update_weights(*fm_update, coef);
}
