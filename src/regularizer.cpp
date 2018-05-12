#include <boost/math/special_functions/sign.hpp>

#include "regularizer.h"


Regularizer::~Regularizer() {}



L2::L2(double C): _C(C), _need_update(false) {}


double L2::get_update(double weight) {
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
    return _C * boost::math::sign(weight);
}


double L1::get_C() const {
    return _C;
}


Regularizer* L1::clone() const {
    Regularizer* clone = new L1(_C);
    return clone;
}