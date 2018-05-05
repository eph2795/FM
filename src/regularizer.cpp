#include <boost/math/special_functions/sign.hpp>

#include "regularizer.h"


Regularizer::~Regularizer() {}

    
void Regularizer::set_update() {
    _need_update = true;
}


L2::L2(double C): _C(C), _need_update(false) {}


double L2::get_update(double weight) {
    if (_need_update) {
        _need_update = false;
        return _C * weight;
    }
    else {
        return 0;
    }
}


L1::L1(double C): _C(C), _need_update(false) {}


double L1::get_update(double weight) {
    if (_need_update) {
        _need_update = false;
        return _C * boost::math::sign(weight);
    }
    else {
        return 0;
    }
}