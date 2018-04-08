#ifndef LOSS_H
#define LOSS_H


#include "data.h"


struct Loss {
    virtual double compute_loss(double prediction, double target) = 0;
    virtual double compute_loss(const Y& prediction, const Y& tartget) = 0;
    virtual double compute_grad(double prediction, double target) = 0;  
};


struct MSE: Loss {
    double compute_loss(double prediction, double target);
    double compute_loss(const Y& prediction, const Y& tartget);
    double compute_grad(double prediction, double target);  
};


#endif