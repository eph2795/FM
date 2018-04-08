#include <iostream>

#include <algorithm>

#include <vector>

#include "data.h"
#include "model.h"
#include "loss.h"
#include "optimization.h"


template<typename T>
void print_vector(const std::vector<T>& v) {
    std::cout << "Vector:" << std::endl;
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl << std::endl;
}


void print_object(const SparseVector& object) {
    for (std::pair<size_t, double> feature: object._items) {
        std::cout << "Col: " << feature.first << ", " << "Val: " << feature.second << "\t";
    }
    std::cout << std::endl;
}



Optimizer::Optimizer(size_t num_epochs, double learning_rate)
        : _num_epochs(num_epochs), _learning_rate(learning_rate)
{}


void Optimizer::train(Model* model, Loss* loss, const X& x, const Y& y) {
    size_t N = x._objects.size();
    // size_t f = model->_w.size();

    std::vector<size_t> objects_order(N);
    std::iota(objects_order.begin(), objects_order.end(), 0);

    // Object w_update;
    for (size_t epoch = 0; epoch < _num_epochs; epoch++) {
        std::random_shuffle(objects_order.begin(), objects_order.end());

        // size_t batch_N = N / _batch_size + (N % _batch_size != 0);
        // std::cout << "Data size: " << N << ", number of batches: " << batch_N << std::endl;
        // size_t i = 0;
        for (size_t obj_idx: objects_order) {
            double prediction = model->predict(x._objects[obj_idx]);
            double coef = -_learning_rate * loss->compute_grad(prediction, y._targets[obj_idx]);
            SparseWeights* m_grad = model->compute_grad(x._objects[obj_idx]);
            model->update_weights(m_grad, coef);
            delete m_grad;
            // std::cout << "Obj idx: " << obj_idx << std::endl;
            // print_object(m_grad);
            // w_update = update(w_update, m_grad, coef);
            // std::cout << "Obj coef: " << coef << std::endl;
            
            // print_object(w_update);
            // break;
            // if (((i + 1) % _batch_size == 0) || (i == N - 1)) {
            //     // print_object(w_update);
            //     update(&(model->_w),  w_update, 1.0 / _batch_size);
            //     w_update._features.resize(0);      
            //     // print_vector(model->_w);
            //     // std::cout << "\n\n\n\n\n";
            //     // break;
            // }
            // // print_vector(model->_w);
            // i += 1;
        }
        // break;
        // print_vector(model->_w);
    }
}