#ifndef MODEL_H
#define MODEL_H

#include <vector>

#include "data.h"
#include "loss.h"
#include "regularizer.h"


struct SparseWeights {
    // virtual SparseWeights& operator+(const SparseWeights* another) = 0;
    virtual ~SparseWeights() = 0;
};


struct LinearSparseWeights: SparseWeights {
    // LinearSparseWeights& operator+(const SparseWeights* another);
    // ~LinearSparseWeights() {};
    double _w0;
    SparseVector _w;
};


struct FMSparseWeights: SparseWeights {
    double _w0;
    SparseVector _w;
    std::vector<std::pair<size_t, std::vector<double>>>  _v;
};



struct Weights {
    virtual ~Weights() = 0;
    virtual void update_weights(const SparseWeights* update, double coef) = 0; 
};


struct LinearWeights: Weights {
    LinearWeights(size_t features_number, Regularizer* regularizer);
    ~LinearWeights();

    void update_weights(const SparseWeights* update, double coef); 

    size_t _features_number;
    double _w0;
    std::vector<double> _w;
    Regularizer* _regularizer;
};


struct FMWeights: Weights {
    FMWeights(size_t features_number, size_t factors_size, Regularizer* regularizer);
    ~FMWeights();

    void update_weights(const SparseWeights* update, double coef);

    size_t _features_number, _factors_size;
    double _w0;
    std::vector<double> _w;
    std::vector<std::vector<double>> _v;
    Regularizer* _regularizer;
};


struct Model {
    virtual ~Model() = 0;

    virtual double predict(const SparseVector& object) = 0;
    virtual Y predict(const X& x) = 0;
    virtual void train(bool state, const std::string& method, size_t num_objects) = 0;
    virtual Weights* get_weights() = 0;
    virtual SparseWeights* compute_grad(const SparseVector& object, double coef) = 0;
    virtual void update_weights(const SparseWeights* update, double coef) = 0;
    virtual void init_als(Loss* loss, const X& x, const Y& y) = 0;
    virtual void als_step(Loss* loss, const X& x, const Y& y) = 0;
    virtual Model* clone() const = 0;
};


struct LinearModel: Model {
    LinearModel(size_t features_number, bool use_offset, Regularizer* regularizer);
    // ~LinearModel();

    double predict(const SparseVector& object); 
    Y predict(const X& x);
    void train(bool state, const std::string& method, size_t num_objects);
    Weights* get_weights();
    SparseWeights* compute_grad(const SparseVector& object, double coef);
    void update_weights(const SparseWeights* update, double coef);
    void init_als(Loss* loss, const X& x, const Y& y);
    void als_step(Loss* loss, const X& x, const Y& y);
    void _linear_als_update();
    Model* clone() const;

    bool _state, _use_offset;
    LinearWeights _weights;
    LinearSparseWeights* _grad;
    std::vector<double> _e;
    // std::vector<double> _x_squares;
    // std::vector<double> _numerators;
};



struct FMModel: Model {
    FMModel(size_t features_number, size_t factors_size, bool use_offset, Regularizer* regularizer);
    // ~FMModel();
    
    double predict(const SparseVector& object); 
    Y predict(const X& x);
    void train(bool state, const std::string& method, size_t num_objects);
    Weights* get_weights();
    SparseWeights* compute_grad(const SparseVector& object, double coef);
    void update_weights(const SparseWeights* update, double coef);
    void init_als(Loss* loss, const X& x, const Y& y);
    void als_step(Loss* loss, const X& x, const Y& y);
    Model* clone() const;

    bool _state, _use_offset;
    FMWeights _weights;
    std::vector<double> _precomputed_sp;
    FMSparseWeights* _grad;
    std::vector<double> _e;
    std::vector<std::vector<double>> _q;
};


#endif
