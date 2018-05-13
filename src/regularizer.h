#ifndef REGULARIZER_H
#define REGULARIZER_H


struct Regularizer {
    virtual ~Regularizer() = 0;
    
    virtual double get_update(double weight) = 0;
    virtual Regularizer* clone() const = 0;
};


struct L2: Regularizer {
    double get_update(double weight);
    Regularizer* clone() const;
};


struct L1: Regularizer {
    double get_update(double weight);
    Regularizer* clone() const;
};


#endif