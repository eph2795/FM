#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <vector>

#include "data.h"


struct Optimizer {
    virtual ~Optimizer() = 0;
    
    virtual void train(Model* model, Loss* loss, const X& x_train, const Y& y_train, 
                       bool use_validation, const X& x_val, const Y& y_val) = 0;
};


struct SGDOptimizer: Optimizer {
    SGDOptimizer(size_t num_epochs, double learning_rate);
    
    void train(Model* model, Loss* loss, const X& x_train, const Y& y_train, 
               bool use_validation, const X& x_val, const Y& y_val); 
    
    size_t _num_epochs;
    double _learning_rate;    
};


#endif 