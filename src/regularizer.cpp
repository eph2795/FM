#include <boost/math/special_functions/sign.hpp>

#include "regularizer.h"


Regularizer::~Regularizer() {}


double L2::get_update(double weight) {
    return weight;
}


Regularizer* L2::clone() const {
    Regularizer* clone = new L2();
    return clone;
}


double L1::get_update(double weight) {
    return boost::math::sign(weight);
}


Regularizer* L1::clone() const {
    Regularizer* clone = new L1();
    return clone;
}