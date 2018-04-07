#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <vector>

#include "data.h"


struct Model { 
    Y predict(const Model& model, const X& x);
    std::vector<double> _w;
};


struct Optimizer {
    Optimizer(size_t num_epochs, double learning_rate);
    void train(Model* model, const X& x, const Y& y); 
    
    size_t _num_epochs;
    double _learning_rate;    
};


// double scalar_product(const Object& object, const std::vector<double>& w);


// double MSE_grad(const Model& model, const X& x, const Y& y, size_t obj_idx);


// Object model_grad(const Model& model, const X& x, size_t obj_idx);


double MSE(const Y& one, const Y& another);


#endif 