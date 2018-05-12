#ifndef REGULARIZER_H
#define REGULARIZER_H


struct Regularizer {
    virtual ~Regularizer() = 0;
    
    virtual double get_update(double weight) = 0;
    void set_update();
    virtual double get_C() const = 0;
    virtual Regularizer* clone() const = 0;

    bool _need_update;
};


struct L2: Regularizer {
    L2(double C);
    double get_update(double weight);
    double get_C() const;
    Regularizer* clone() const;

    double _C;
    bool _need_update;
};


struct L1: Regularizer {
    L1(double C);
    double get_update(double weight);
    double get_C() const;
    Regularizer* clone() const;

    double _C;
    bool _need_update;
};


#endif