#include <iostream>
#include <fstream>
#include <string.h>
#include <cmath>
#include <algorithm>
#include <random>
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


Weights::~Weights() {}


LinearWeights::LinearWeights(size_t features_number, 
                             double C0, double Cw, Regularizer* regularizer)
        : _features_number(features_number), _w0(0), _w(features_number, 0), _C0(C0), _Cw(Cw), _regularizer(regularizer)
{}


LinearWeights::~LinearWeights() {
    delete _regularizer;
}


inline void LinearWeights::update_weights(const SparseWeights* update, double coef) {
    const LinearSparseWeights* linear_update = dynamic_cast<const LinearSparseWeights*>(update);
    _w0 += coef * (linear_update->_w0 + _C0 * _regularizer->get_update(_w0));
    for (std::pair<size_t, double> item: linear_update->_w._items) {
        _w[item.first] += coef * (item.second + _Cw *_regularizer->get_update(_w[item.first]));
    }
}


FMWeights::FMWeights(size_t features_number, size_t factors_size, 
                     double C0, double Cw, const std::vector<double>& Cv, Regularizer* regularizer)
        : _features_number(features_number), _factors_size(factors_size), _w0(0)
        , _w(features_number, 0), _v(features_number, std::vector<double>(factors_size))
        , _C0(C0), _Cw(Cw), _Cv(Cv), _regularizer(regularizer)
{
     std::default_random_engine generator;
     std::normal_distribution<double> weights_sampler(0, 0.001);
     for (size_t i = 0; i < features_number; i++) {
        for (size_t j = 0; j < factors_size; j++) {
            _v[i][j] = weights_sampler(generator);
        }
     }
}


FMWeights::~FMWeights() {
    delete _regularizer;
}


inline void FMWeights::update_weights(const SparseWeights* update, double coef) {
    const FMSparseWeights* fm_update = dynamic_cast<const FMSparseWeights*>(update);
    _w0 += coef * (fm_update->_w0 + _C0 * _regularizer->get_update(_w0));
    for (std::pair<size_t, double> item: fm_update->_w._items) {
        _w[item.first] += coef * (item.second + _Cw * _regularizer->get_update(_w[item.first]));
    }
    for (std::pair<size_t, std::vector<double>> factors: fm_update->_v) {
        for (size_t factor_idx = 0; factor_idx < _factors_size; factor_idx++) {
            _v[factors.first][factor_idx] += coef * (factors.second[factor_idx] 
                + _Cv[factor_idx] * _regularizer->get_update(_v[factors.first][factor_idx]));
        }
    }
}


Model::~Model() {}


LinearModel::LinearModel(size_t features_number, bool use_offset, 
                         double C0, double Cw, Regularizer* regularizer)
        : _state(false), _use_offset(use_offset), _weights(features_number, C0, Cw, regularizer), _e(0)
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
    prediction._targets.resize(x._objects_number);
    for (size_t obj_idx = 0; obj_idx < x._objects.size(); obj_idx++) {
        prediction._targets[obj_idx] = predict(x._objects[obj_idx]);
    }
    return prediction;
}


void LinearModel::train_sgd(bool state) {
    if (state and not _state) {
        _state = true;
        _grad = new LinearSparseWeights();
    } else if (not state and _state) {
         _state = false;
        delete _grad;
    }
}


Weights* LinearModel::get_weights() {
    return &_weights;
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


inline void LinearModel::update_reg(const SparseVector& object, const SparseWeights* update, double coef) {
    _weights._C0 = std::max(0.0, _weights._C0 - 2 * coef * _weights._w0);

    double Cw = 0;
    for (const auto& item: object._items) {
        Cw += _weights._w[item.first] * item.second;
    }

    _weights._Cw = std::max(0.0, _weights._Cw - 2 * coef * Cw);
}


void LinearModel::init_als(Loss* loss, const X& x, const X& x_csr, const Y& y) {
    _e.resize(x._objects_number);
    for (size_t object_idx = 0; object_idx < x._objects_number; object_idx++) {
        _e[object_idx] = predict(x_csr._objects[object_idx]) - y._targets[object_idx];
    } 
}


void LinearModel::als_step(Loss* loss, const X& x, const Y& y) {
    size_t M = x._features_number; // number of features
    size_t N = x._objects_number; // number of objects  
    size_t object_idx;
    double C0 = _weights._C0 * N, Cw = _weights._Cw * N;
    double dw, numerator, squares, feature_value, e_sum = 0;

    for (size_t object_idx = 0; object_idx < N; object_idx++) {
        e_sum += _e[object_idx];
    }

    dw = -(e_sum - N * _weights._w0) / (N + C0) - _weights._w0;
    for (size_t object_idx = 0; object_idx < N; object_idx++) {
        _e[object_idx] += dw;
    }
    _weights._w0 += dw;


    std::vector<size_t> features_order(M);
    std::iota(features_order.begin(), features_order.end(), 0);
    std::random_shuffle(features_order.begin(), features_order.end());

    for (size_t feature_idx: features_order) {
        numerator = 0;
        squares = 0;
        for (const auto& item: x._objects[feature_idx]._items) {
            object_idx = item.first;
            feature_value = item.second;
            numerator += feature_value * (_e[object_idx] - _weights._w[feature_idx] * feature_value);
            squares += std::pow(feature_value, 2);
        }
        dw = -numerator / (squares + Cw) - _weights._w[feature_idx];
        for (const auto& item: x._objects[feature_idx]._items) {
            object_idx = item.first;
            feature_value = item.second;
            _e[object_idx] += dw * feature_value;
        }
        _weights._w[feature_idx] += dw;
    }
}


Model* LinearModel::clone() const {
    Regularizer* regularizer = _weights._regularizer->clone();
    LinearModel* clone = new LinearModel(_weights._features_number, _use_offset, _weights._C0, _weights._Cw, regularizer);
    clone->_weights = _weights;
    return clone;
}


void LinearModel::dump(const std::string& file_name) const {
    std::ofstream file;
    file.open(file_name.c_str(), std::ios::out | std::ios::binary);

    if (file.is_open()) {
        file.write((char*)&_use_offset, sizeof(bool));
        file.write((char*)&_weights._features_number, sizeof(double));
        file.write((char*)&_weights._w0, sizeof(double));
        file.write((char*)_weights._w.data(), sizeof(double) * _weights._w.size());
        file.write((char*)&_weights._C0, sizeof(double));
        file.write((char*)&_weights._Cw, sizeof(double));
        file.write((char*)_weights._regularizer, sizeof(Regularizer));
        file.close();
    } else {
        std::cout << "Unable to open model file! Terminated." << std::endl;
        throw;
    }
}


void LinearModel::load(const std::string& file_name) const {
    std::ifstream file;
    file.open(file_name.c_str(), std::ios::in | std::ios::binary);
    
    if (file.is_open()) {
        file.read((char*)&_use_offset, sizeof(bool));
        file.read((char*)&_weights._features_number, sizeof(double));
        file.read((char*)&_weights._w0, sizeof(double));
        file.read((char*)_weights._w.data(), sizeof(double) * _weights._features_number);
        file.read((char*)&_weights._C0, sizeof(double));
        file.read((char*)&_weights._Cw, sizeof(double));
        file.read((char*)_weights._regularizer, sizeof(Regularizer));
        file.close();   
    } else {
        std::cout << "Unable to open model file! Terminated." << std::endl;
        throw;
    }
}


FMModel::FMModel(size_t features_number, size_t factors_size, bool use_offset, 
                 double C0, double Cw, const std::vector<double>& Cv, Regularizer* regularizer)
        : _state(false), _use_offset(use_offset)
        , _weights(features_number, factors_size, C0, Cw, Cv, regularizer)
        , _precomputed_sp(factors_size), _e(0), _q(0, std::vector<double>(0))
{}


// FMModel::~FMModel() {
//     delete _grad;
// }


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


void FMModel::train_sgd(bool state) {
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
 
    _grad->_v.resize(0);
    for (auto feature: object._items) {
        std::vector<double> new_grad(_weights._factors_size);
        for (size_t factor_num = 0; factor_num < _weights._factors_size; factor_num++) {
            double term = _precomputed_sp[factor_num];
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


void FMModel::update_reg(const SparseVector& object, const SparseWeights* update, double coef) {
    const FMSparseWeights* fm_update = dynamic_cast<const FMSparseWeights*>(update);
    _weights._C0 = std::max(0.0, _weights._C0 - 2 * coef * _weights._w0);

    double Cw = 0;
    for (const auto& item: object._items) {
        Cw += _weights._w[item.first] * item.second;
    }
    _weights._Cw = std::max(0.0, _weights._Cw - 2 * coef * Cw);

    std::vector<double> Cv(_weights._factors_size, 0);
    for (size_t factor_idx = 0; factor_idx < _weights._factors_size; factor_idx++) {
        double cur_term, next_term;
        double xv_cur = 0, xv_next = 0, xv_quad = 0;
        auto obj_iter = object._items.begin();
        auto upd_iter = fm_update->_v.begin();
        for (; (obj_iter != object._items.end()) and (upd_iter != fm_update->_v.end()); ) {
            if (obj_iter->first > upd_iter->first) {
                upd_iter++;
            } else {            
                cur_term = _weights._v[obj_iter->first][factor_idx] * obj_iter->second;
                xv_cur += cur_term;
                if (obj_iter->first == upd_iter->first) {
                    next_term = (_weights._v[obj_iter->first][factor_idx] + upd_iter->second[factor_idx]) * obj_iter->second;
                    xv_next += next_term;
                    xv_quad += cur_term * next_term;
                    upd_iter++;
                }
                obj_iter++;  
            } 
        }
        _weights._Cv[factor_idx] = std::max(0.0, _weights._Cv[factor_idx] - 2 * coef * (xv_cur * xv_next - xv_quad));
    }   
}


void FMModel::init_als(Loss* loss, const X& x, const X& x_csr, const Y& y) {
    _e.resize(x._objects_number);
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
    size_t object_idx;  
    double C0 = _weights._C0 * N, Cw = _weights._Cw * N;
    std::vector<double> Cv = _weights._Cv;
    for (size_t factor_idx = 0; factor_idx < Cv.size(); factor_idx++) {
        Cv[factor_idx] *= N;
    }    
    double dw, numerator, squares, h, feature_value, e_sum = 0;

    for (size_t object_idx = 0; object_idx < N; object_idx++) {
        e_sum += _e[object_idx];
    }

    dw = -(e_sum - N * _weights._w0) / (N + C0) - _weights._w0;
    for (size_t object_idx = 0; object_idx < N; object_idx++) {
        _e[object_idx] += dw;
    }
    _weights._w0 += dw;

    std::vector<size_t> features_order(M);
    std::iota(features_order.begin(), features_order.end(), 0);
    std::random_shuffle(features_order.begin(), features_order.end());

    for (size_t feature_idx: features_order) {
        numerator = 0;
        squares = 0;
        for (const auto& item: x._objects[feature_idx]._items) {
            object_idx = item.first;
            feature_value = item.second;
            numerator += feature_value * (_e[object_idx] - _weights._w[feature_idx] * feature_value);
            squares += std::pow(feature_value, 2);
        }
        dw = -numerator / (squares + Cw) - _weights._w[feature_idx];
        for (const auto& item: x._objects[feature_idx]._items) {
            object_idx = item.first;
            feature_value = item.second;
            _e[object_idx] += dw * feature_value;
        }
        _weights._w[feature_idx] += dw;
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
            
            dw = -numerator / (squares + Cv[factor_idx]) - _weights._v[feature_idx][factor_idx];
            for (const auto& item: x._objects[feature_idx]._items) {
                object_idx = item.first;
                feature_value = item.second;
                
                h = feature_value * (_q[object_idx][factor_idx] - _weights._v[feature_idx][factor_idx] * feature_value);
                _e[object_idx] += dw * h;
                _q[object_idx][factor_idx] += dw * feature_value;
            }
            _weights._v[feature_idx][factor_idx] += dw;
        }
    }
}


Model* FMModel::clone() const {
    Regularizer* regularizer = _weights._regularizer->clone();
    FMModel* clone = new FMModel(_weights._features_number, _weights._factors_size, _use_offset, 
                                _weights._C0, _weights._Cw, _weights._Cv, regularizer);
    clone->_weights = _weights;
    return clone;  
}


void FMModel::dump(const std::string& file_name) const {
    std::ofstream file;
    file.open(file_name.c_str(), std::ios::out | std::ios::binary);

    if (file.is_open()) {
        file.write((char*)&_use_offset, sizeof(bool));
        file.write((char*)&_weights._features_number, sizeof(double));
        file.write((char*)&_weights._factors_size, sizeof(double));
        file.write((char*)&_weights._w0, sizeof(double));
        file.write((char*)_weights._w.data(), sizeof(double) * _weights._w.size());
        file.write((char*)&_weights._C0, sizeof(double));
        file.write((char*)&_weights._Cw, sizeof(double));
        file.write((char*)_weights._Cv.data(), sizeof(double) * _weights._Cv.size());
        file.write((char*)_weights._regularizer, sizeof(Regularizer));
        for (const auto& v: _weights._v) {
            file.write((char*)v.data(), sizeof(double) * v.size());
        }
        file.close();
    } else {
        std::cout << "Unable to open model file! Terminated." << std::endl;
        throw;
    }
}


void FMModel::load(const std::string& file_name) const {
    std::ifstream file;
    file.open(file_name.c_str(), std::ios::in | std::ios::binary);
    
    if (file.is_open()) {
        file.read((char*)&_use_offset, sizeof(bool));
        file.read((char*)&_weights._features_number, sizeof(double));
        file.read((char*)&_weights._factors_size, sizeof(double));
        file.read((char*)&_weights._w0, sizeof(double));
        file.read((char*)_weights._w.data(), sizeof(double) * _weights._features_number);
        file.read((char*)&_weights._C0, sizeof(double));
        file.read((char*)&_weights._Cw, sizeof(double));
        file.read((char*)_weights._Cv.data(), sizeof(double) * _weights._Cv.size());
        file.read((char*)_weights._regularizer, sizeof(Regularizer));
        for (auto& v: _weights._v) {
            file.read((char*)v.data(), sizeof(double) * _weights._factors_size);
        }
        file.close();   
    } else {
        std::cout << "Unable to open model file! Terminated." << std::endl;
        throw;
    }
}
