#ifndef MODEL_H
#define MODEL_H

#include <vector>

#include "data.h"


struct SparseWeights {
    // virtual SparseWeights& operator+(const SparseWeights* another) = 0;
    virtual ~SparseWeights() = 0;
};


// SparseWeights::~SparseWeights() {};


// struct Weights { 
//     virtual SparseWeights* compute_grad(const SparseVector& object) = 0;
//     virtual void update_weights(const SparseWeights* another, double coef) = 0; 
// };


struct LinearSparseWeights: SparseWeights {
    // LinearSparseWeights& operator+(const SparseWeights* another);
    // ~LinearSparseWeights() {};
    double _w0;
    SparseVector _w;
};


struct LinearWeights {
    LinearWeights(size_t features_number);
    inline void update_weights(const LinearSparseWeights& another, double coef); 
    double _w0;
    std::vector<double> _w;
};


struct Model {
    inline virtual double predict(const SparseVector& object) = 0;
    virtual Y predict(const X& x) = 0;
    inline virtual SparseWeights* compute_grad(const SparseVector& object) = 0;
    inline virtual void update_weights(const SparseWeights* update, double coef) = 0;
};


struct LinearModel: Model {
    LinearModel(size_t features_number, bool use_offset);
    inline double predict(const SparseVector& object); 
    Y predict(const X& x);
    inline SparseWeights* compute_grad(const SparseVector& object);
    inline void update_weights(const SparseWeights* update, double coef);

    bool _use_offset;
    LinearWeights _weights;
};


#endif
