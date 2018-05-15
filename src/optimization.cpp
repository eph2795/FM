#include <iostream>
#include <ctime>
#include <algorithm>
#include <vector>

#include <omp.h>

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


 Optimizer::~Optimizer() {};


SGDOptimizer::SGDOptimizer(size_t num_epochs, double learning_rate)
        : _num_epochs(num_epochs), _learning_rate(learning_rate)
{}


void SGDOptimizer::train(Model* model, Loss* loss, const X& x_train, const Y& y_train, 
        bool use_validation, const X& x_val, const Y& y_val, bool adaptive_reg) {

    size_t N = x_train._objects.size();
    std::vector<size_t> objects_order(N);
    std::iota(objects_order.begin(), objects_order.end(), 0);

    std::default_random_engine generator;
    size_t val_size = 0;
    if (use_validation and adaptive_reg) {
        val_size = x_val._objects_number;
    }
    std::uniform_int_distribution<size_t> val_sampler(0, val_size);

    size_t c_iter = 1, c_sample = 1;
    double average_loss = 0;

    model->train_sgd(true);
    for (size_t epoch = 0; epoch < _num_epochs; epoch++) {
        std::random_shuffle(objects_order.begin(), objects_order.end());

        double start, finish;
        start = clock();

        double prediction, coef, obj_loss, train_loss = 0;
        size_t obj_idx;

     
        for (size_t i = 0; i < N; i++) {
            obj_idx = objects_order[i];
            prediction = model->predict(x_train._objects[obj_idx]);
            obj_loss = loss->compute_loss(prediction, y_train._targets[obj_idx]);
            train_loss += obj_loss;
            coef = loss->compute_grad(prediction, y_train._targets[obj_idx]);
            SparseWeights* model_grad = model->compute_grad(x_train._objects[obj_idx], coef);

            if (use_validation and adaptive_reg) {
                size_t sampled_idx = val_sampler(generator);
                model->update_reg(x_train._objects[sampled_idx], model_grad, -std::pow(_learning_rate, 2) * coef);
            }

            model->update_weights(model_grad, -_learning_rate);

            average_loss += obj_loss / std::pow(2, c_iter);
            if (std::pow(2, c_iter) == c_sample) {
                std::cout << "Passed samples: " << c_sample << ", loss: " << average_loss << std::endl;
                average_loss = 0;
                c_sample = 0;
                c_iter += 1;
            }
            c_sample += 1;
        }
        train_loss /= N;
        
        double val_loss;
        if (use_validation) {
            Y val_prediction = model->predict(x_val);
            val_loss = loss->compute_loss(val_prediction, y_val);
        }

        finish = clock();
        std::cout << "Epoch " << epoch << " finished.\n \tTrain loss: " << train_loss << ",\n";
        if (use_validation) {
            std::cout << "\tvalidation loss: " << val_loss << ",\n";
        }
        // if (adaptive_reg) {
        //     const LinearModel* linear_model = dynamic_cast<const LinearModel*>(model);
        //     std::cout << "\tC0: " << linear_model->_weights._C0 << ", Cw: " << linear_model->_weights._Cw << ",\n";
        // }
        std::cout << "\telapsed time: " << double(finish - start) / CLOCKS_PER_SEC << "." << std::endl;
    }
    model->train_sgd(false);
}


ALSOptimizer::ALSOptimizer(size_t num_epochs): _num_epochs(num_epochs) {}


void ALSOptimizer::train(Model* model, Loss* loss, const X& x_train_csr, const Y& y_train, 
        bool use_validation, const X& x_val_csr, const Y& y_val, bool adaptive_reg) {
    X x_train = x_train_csr.to_csc();
    X x_val = x_val_csr.to_csc();

    model->init_als(loss, x_train, x_train_csr, y_train);

    for (size_t epoch = 0; epoch < _num_epochs; epoch++) {
        double start, finish;

        start = clock();

        model->als_step(loss, x_train, y_train);

        Y train_prediction = model->predict(x_train_csr);
        double train_mse = loss->compute_loss(train_prediction, y_train);

        double val_mse;
        if (use_validation) {
            Y val_prediction = model->predict(x_val_csr);
            val_mse = loss->compute_loss(val_prediction, y_val);
        }

        finish = clock();
        std::cout << "Epoch " << epoch << " finished.\n \tTrain loss: " << train_mse << ",\n";
        if (use_validation) {
            std::cout << "\tvalidation loss: " << val_mse << ",\n";
        }
        std::cout << "\telapsed time: " << double(finish - start) / CLOCKS_PER_SEC << "." << std::endl;
        
    }
}


// SVRGOptimizer::SVRGOptimizer(size_t num_epochs, double learning_rate, size_t update_frequency)
//         : _num_epochs(num_epochs), _learning_rate(learning_rate), _update_frequency(update_frequency)
// {}


// void SVRGOptimizer::train(Model* model, Loss* loss, const X& x_train, const Y& y_train, 
//         bool use_validation, const X& x_val, const Y& y_val) {
//     size_t N = x_train._objects.size();
//     std::default_random_engine generator;
//     std::uniform_int_distribution<size_t> object_sampler(0, N);
//     std::uniform_int_distribution<size_t> gradient_sampler(0, _update_frequency);
    
//     model->train_sgd(true);
//     for (size_t epoch = 0; epoch < _num_epochs; epoch++) {
    
//         double start, finish, train_mse = 0;
//         start = clock();
        
//         size_t iter_after_update = 0;
//         for (size_t i = 0; i < N; i++) {
//             if (iter_after_update == 0) {
//                 Model* clone = model->clone();
//                 mu = 
//             }
//             double prediction = model->predict(x_train._objects[obj_idx]);
//             train_mse += loss->compute_loss(prediction, y_train._targets[obj_idx]);
//             double coef = loss->compute_grad(prediction, y_train._targets[obj_idx]);
//             SparseWeights* model_grad = model->compute_grad(x_train._objects[obj_idx], coef);
//             model->update_weights(model_grad, -_learning_rate);
//         }
//         train_mse /= N;
        
//         double val_mse;
//         if (use_validation) {
//             Y val_prediction = model->predict(x_val);
//             val_mse = loss->compute_loss(val_prediction, y_val);
//         }

//         finish = clock();
//         std::cout << "Epoch " << epoch << " finished.\n \tTrain loss: " << train_mse << ",\n";
//         if (use_validation) {
//             std::cout << "\tvalidation loss: " << val_mse << ",\n";
//         }
//         std::cout << "\telapsed time: " << double(finish - start) / CLOCKS_PER_SEC << "." << std::endl;;
//     }
//     model->train_sgd(false);
// }