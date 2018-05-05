#include <iostream>
#include <string.h>

#include <string>
#include <vector>

#include <ctime>

#include "data.h"
#include "model.h"
#include "loss.h"
#include "optimization.h"
#include "regularizer.h"        


void parse_arguments(int argc, char** argv, 
        std::string* train_file, std::string* test_file, std::string* model_type, std::string* loss_type, 
        size_t* factors_size, bool* use_offset, size_t* num_epochs, double* learning_rate, std::string* reg_type, double* C) {
    for (size_t i = 1; i < static_cast<size_t>(argc); i++) {
        try {
            if (strcmp(argv[i], "--data") == 0) {
                i += 1;
                *train_file = std::string(argv[i]); 
            } else if (strcmp(argv[i], "--test") == 0) {
                i += 1;
                *test_file = std::string(argv[i]);
            } else if (strcmp(argv[i], "--model") == 0) {
                i += 1;
                *model_type = std::string(argv[i]);
            } else if (strcmp(argv[i], "--loss") == 0) {
                i += 1;
                *loss_type = std::string(argv[i]);
            } else if (strcmp(argv[i], "--factors_size") == 0) {
                i += 1;
                *factors_size = std::stoul(argv[i]);
            } else if (strcmp(argv[i], "--use_offset") == 0) {
                *use_offset = true;
            } else if (strcmp(argv[i], "--passes") == 0) {
                i += 1;
                *num_epochs = std::stoul(argv[i]);
            } else if (strcmp(argv[i], "--learning_rate") == 0) {
                i += 1;
                *learning_rate = std::stod(argv[i]);
            } else if (strcmp(argv[i], "--reg_type") == 0) {
                i += 1;
                *reg_type = std::string(argv[i]);           
            } else if (strcmp(argv[i], "-C") == 0) {
                i += 1;
                *C = std::stod(argv[i]);
            }
        } 
        catch (std::invalid_argument) {
            throw std::invalid_argument("Wrong arguments format!");
        }
    }
}


Regularizer* create_regularizer(const std::string& reg_type, double C) {
    Regularizer* regularizer;
    if (strcmp(reg_type.c_str(), "L1") == 0) {
        std::cout << "Using L1 regularizer with const=" << C << "." <<std::endl;
        regularizer = new L1(C);
    } else if (strcmp(reg_type.c_str(), "L2") == 0) {
        std::cout << "Using L2 regularizer with const=" << C << "." <<std::endl;
        regularizer = new L2(C);
    } else {
        std::cout << "Wrong regularizer name! Terminated." << std::endl;
        throw;
    }
    return regularizer;
}


Model* create_model(const std::string& model_type, size_t features_number, size_t factors_size, bool use_offset,
                    const std::string& reg_type, double C) {
    Regularizer* regularizer = create_regularizer(reg_type, C);
    Model* model;
    if (strcmp(model_type.c_str(), "linear") == 0) {
        std::cout << "Using linear model." << std::endl;
        model = new LinearModel(features_number, use_offset, regularizer);
    } else if (strcmp(model_type.c_str(), "fm") == 0) {
        std::cout << "Using FM model with " << factors_size << " factors." << std::endl;
        model = new FMModel(features_number, factors_size, use_offset, regularizer);
    } else {
        std::cout << "Wrong model name! Terminated." << std::endl;
        throw;
    }
    return model;
}


Loss* create_loss(const std::string& loss_type) {
    Loss* loss;
    if (strcmp(loss_type.c_str(), "mse") == 0) {
        std::cout << "Using MSE loss." << std::endl;
        loss = new MSE();
    } else if (strcmp(loss_type.c_str(), "logistic") == 0) {
        std::cout << "Using logistic loss." << std::endl;
        loss = new Logistic();
    } else {
        std::cout << "Wrong loss name! Terminated." << std::endl;
        throw;
    }
    return loss;
}



int main(int argc, char** argv) {
    std::string train_file("../../datasets/train_data.vw");  
    std::string test_file("../../datasets/test_data.vw");  
    std::string model_type("linear");
    std::string loss_type("mse");
    std::string reg_type("L2");

    size_t factors_size = 10;
    double learning_rate = 1e-3;
    double C=1e-3;
    size_t num_epochs = 10;
    bool use_offset = false;
    parse_arguments(argc, argv, &train_file, &test_file, &model_type, &loss_type, &factors_size, &use_offset, &num_epochs, &learning_rate, &reg_type, &C);
    clock_t start, finish;

    std::cout << "Train data file: " << train_file << std::endl;
    std::cout << "Test data file: " << test_file << std::endl;
    std::cout << std::endl;

    std::cout << "Start to preprocessing train file..." << std::endl;
    start = clock();
    DataReader data_reader;
    data_reader.get_columns_info(train_file);
    finish = clock();
    std::cout << "Train file preprocessed! Elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << std::endl;

    std::cout << "Start to reading train data.." << std::endl;
    start = clock();
    X x_train;
    Y y_train;  
    data_reader.fill_with_data(train_file, &x_train, &y_train);
    finish = clock();
    std::cout << "Reading finished! Elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << std::endl;

    Model* model = create_model(model_type, data_reader._features_number, factors_size, use_offset, reg_type, C);
    Loss* loss = create_loss(loss_type);
    std::cout << "Passes number: " << num_epochs << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << std::endl;

    std::cout << "Start to train model..." << std::endl;
    start = clock();
    Optimizer optimizer(num_epochs, learning_rate);
    optimizer.train(model, loss, x_train, y_train);
    Y train_prediction = model->predict(x_train);
    double train_mse = loss->compute_loss(train_prediction, y_train); 
    finish = clock();
    std::cout << "Training finished! Elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "Loss: " << train_mse << std::endl;
    std::cout << std::endl;

    std::cout << "Start to predict on test data..." << std::endl;
    start = clock();
    X x_test;
    Y y_test;
    data_reader.fill_with_data(test_file, &x_test, &y_test);
    // for (size_t i = 0; i < x_test._objects.size(); i++) {
    //     for (size_t j = 0; j < x_test._objects[i]._features.size(); j++) {
    //         std::cout << "Idx: " << x_test._objects[i]._features[j].idx << ", value: " << x_test._objects[i]._features[j].value << "\t";
    //     }
    //     std::cout << "\t target: "  << y_test._targets[i] << std::endl;
    // }
    // print_vector(model._w);
    Y test_prediction = model->predict(x_test);
    double test_mse = loss->compute_loss(test_prediction, y_test);
    finish = clock();
    std::cout << "Prediction finished! Elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "Loss: " << test_mse << std::endl;

    delete model;
    delete loss;
    return 0;
}
