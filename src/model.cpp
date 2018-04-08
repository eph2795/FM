#include <utility>
#include <vector>

#include "data.h"
#include "model.h"


SparseWeights::~SparseWeights() {};

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


LinearWeights::LinearWeights(size_t features_number): _w(features_number) {}


inline void LinearWeights::update_weights(const LinearSparseWeights& update, double coef) {
    _w0 += coef * update._w0;
    for (std::pair<size_t, double> item: update._w._items) {
        _w[item.first] += coef * item.second;
    }
}


LinearModel::LinearModel(size_t features_number, bool use_offset)
        : _use_offset(use_offset), _weights(features_number)
{}


inline double LinearModel::predict(const SparseVector& object) {
    double prediction = 0;
    for (std::pair<size_t, double> feature: object._items) {
        prediction += _weights._w[feature.first] * feature.second;
    }
    if (_use_offset) {
        prediction += _weights._w0;
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


inline SparseWeights* LinearModel::compute_grad(const SparseVector& object) {
    LinearSparseWeights* m_grad = new LinearSparseWeights;
    m_grad->_w0 = 1;
    m_grad->_w = object;
    return m_grad;
}


inline void LinearModel::update_weights(const SparseWeights* update, double coef) {
    const LinearSparseWeights* linear_update = dynamic_cast<const LinearSparseWeights*>(update);
    _weights.update_weights(*linear_update, coef);
    // _weights._w0 += coef * linear_update->_w0;
    // for (std::pair<size_t, double> feature: linear_update->_w._items) {
    //     _weights._w[feature.first] += coef * feature.second;
    // }
}