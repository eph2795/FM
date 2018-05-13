#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <vector>

#include "data.h"


struct Optimizer {
    virtual ~Optimizer() = 0;
    
    virtual void train(Model* model, Loss* loss, const X& x_train, const Y& y_train, 
                       bool use_validation, const X& x_val, const Y& y_val, bool adaptive_reg=false) = 0;
};


struct SGDOptimizer: Optimizer {
    SGDOptimizer(size_t num_epochs, double learning_rate);
    
    void train(Model* model, Loss* loss, const X& x_train, const Y& y_train, 
               bool use_validation, const X& x_val, const Y& y_val, bool adaptive_reg=false); 
    
    size_t _num_epochs;
    double _learning_rate;    
};


struct ALSOptimizer: Optimizer {
    ALSOptimizer(size_t num_epochs);
    
    void train(Model* model, Loss* loss, const X& x_train, const Y& y_train, 
               bool use_validation, const X& x_val, const Y& y_val, bool adaptive_reg=false); 
    
    size_t _num_epochs;
};


// struct SVRGOptimizer: Optimizer {
//     SVRGOptimizer(size_t num_epochs, double learning_rate, size_t update_frequency);
    
//     void train(Model* model, Loss* loss, const X& x_train, const Y& y_train, 
//                bool use_validation, const X& x_val, const Y& y_val); 
    
//     size_t _num_epochs;
//     double _learning_rate;    
//     size_t _update_frequency;
// };

#endif 