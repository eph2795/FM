#include <boost/math/special_functions/sign.hpp>

#include "regularizer.h"


Regularizer::~Regularizer() {}

    
void Regularizer::set_update() {
    _need_update = true;
}


L2::L2(double C): _C(C), _need_update(false) {}


double L2::get_update(double weight) {
    // if (_need_update) {
    //     _need_update = false;
    //     return _C * weight;
    // }
    // else {
    //     return 0;
    // }
    return _C * weight;
}


double L2::get_C() const {
    return _C;
}


Regularizer* L2::clone() const {
    Regularizer* clone = new L2(_C);
    return clone;
}

L1::L1(double C): _C(C), _need_update(false) {}


double L1::get_update(double weight) {
    // if (_need_update) {
    //     _need_update = false;
    //     return _C * boost::math::sign(weight);
    // }
    // else {
    //     return 0;
    // }
    return _C * boost::math::sign(weight);
}


double L1::get_C() const {
    return _C;
}


Regularizer* L1::clone() const {
    Regularizer* clone = new L1(_C);
    return clone;
}