#include <iostream>
#include <ctime>
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


void Optimizer::train(Model* model, Loss* loss, const X& x_train, const Y& y_train, 
        bool use_validation, const X& x_val, const Y& y_val) {
    size_t N = x_train._objects.size();
    // size_t f = model->_w.size();

    std::vector<size_t> objects_order(N);
    std::iota(objects_order.begin(), objects_order.end(), 0);

    model->train(true);
    // Object w_update;
    for (size_t epoch = 0; epoch < _num_epochs; epoch++) {
        std::random_shuffle(objects_order.begin(), objects_order.end());

        // size_t batch_N = N / _batch_size + (N % _batch_size != 0);
        // std::cout << "Data size: " << N << ", number of batches: " << batch_N << std::endl;
        // size_t i = 0;
        double start, finish, train_mse = 0;
        start = clock();
        for (size_t obj_idx: objects_order) {
            double prediction = model->predict(x_train._objects[obj_idx]);
            train_mse += loss->compute_loss(prediction, y_train._targets[obj_idx]);
            double coef = loss->compute_grad(prediction, y_train._targets[obj_idx]);
            SparseWeights* model_grad = model->compute_grad(x_train._objects[obj_idx], coef);
            model->update_weights(model_grad, -_learning_rate);
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
        train_mse /= N;
        
        double val_mse;
        if (use_validation) {
            Y val_prediction = model->predict(x_val);
            val_mse = loss->compute_loss(val_prediction, y_val);
        }

        finish = clock();
        std::cout << "Epoch " << epoch << " finished. Train loss: " << train_mse << ", ";
        if (use_validation) {
            std::cout << "validation loss: " << val_mse << ", ";
        }
        std::cout << "elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
        // break;
        // print_vector(model->_w);
    }
    model->train(false);
}