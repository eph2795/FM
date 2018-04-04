#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <vector>

#include "data.h"


struct Model { 
    Y predict(const Model& model, const X& x);
    std::vector<double> _w;
};


struct Optimizer {
    Optimizer(size_t num_epochs, double learning_rate, size_t batch_size);
    void train(Model* model, const X& x, const Y& y); 
    
    size_t _num_epochs;
    double _learning_rate;
    size_t _batch_size;
    
};


double scalar_product(const Object& object, const std::vector<double>& w);


double MSE_grad(const Model& model, const X& x, const Y& y, size_t obj_idx);


std::vector<double> model_grad(const Model& model, const X& x, size_t obj_idx);


double MSE(const Y& one, const Y& another);


#endif 