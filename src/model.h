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


struct FMSparseWeights: SparseWeights {
    double _w0;
    SparseVector _w;
    std::map<size_t, std::vector<double>> _v;
};


struct LinearWeights {
    LinearWeights(size_t features_number);
    
    void update_weights(const LinearSparseWeights& update, double coef); 

    size_t _features_number;
    double _w0;
    std::vector<double> _w;
};


struct FMWeights {
    FMWeights(size_t features_number, size_t factors_size);

    void update_weights(const FMSparseWeights& update, double coef);

    size_t _features_number, _factors_size;
    double _w0;
    std::vector<double> _w;
    std::vector<std::vector<double>> _v;
};


struct Model {
    virtual ~Model() = 0;

    virtual double predict(const SparseVector& object) = 0;
    virtual Y predict(const X& x) = 0;
    virtual SparseWeights* compute_grad(const SparseVector& object) = 0;
    virtual void update_weights(const SparseWeights* update, double coef) = 0;
};


struct LinearModel: Model {
    LinearModel(size_t features_number, bool use_offset);

    double predict(const SparseVector& object); 
    Y predict(const X& x);
    SparseWeights* compute_grad(const SparseVector& object);
    void update_weights(const SparseWeights* update, double coef);

    bool _use_offset;
    // size_t _features_number;
    LinearWeights _weights;
};



struct FMModel: Model {
    FMModel(size_t features_number, size_t factors_size, bool use_offset);

    double predict(const SparseVector& object); 
    Y predict(const X& x);
    SparseWeights* compute_grad(const SparseVector& object);
    void update_weights(const SparseWeights* update, double coef);

    bool _use_offset;
    // size_t _features_number, _factors_size;
    FMWeights _weights;
    std::vector<double> _precomputed_sp;
};


#endif
