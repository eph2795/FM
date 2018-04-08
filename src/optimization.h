#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <vector>

#include "data.h"


struct Optimizer {
    Optimizer(size_t num_epochs, double learning_rate);
    void train(Model* model, Loss* loss, const X& x, const Y& y); 
    
    size_t _num_epochs;
    double _learning_rate;    
};


#endif 