#include <iostream>
#include <string.h>

#include <string>
#include <vector>

#include <ctime>

#include "data.h"
#include "model.h"
#include "loss.h"
#include "optimization.h"
        

void parse_arguments(int argc, char** argv, 
        std::string* train_file, std::string* test_file, std::string* model_type, 
        size_t* factors_size, bool* use_offset, size_t* num_epochs, double* learning_rate) {
    for (size_t i = 1; i < argc; i++) {
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
            }
        } 
        catch (std::invalid_argument) {
            throw std::invalid_argument("Wrong arguments format!");
        }
    }
}


Model* create_model(const std::string& model_type, size_t features_number, size_t factors_size, bool use_offset) {
    Model* model;
    if (strcmp(model_type.c_str(), "linear") == 0) {
        std::cout << "Using linear model." << std::endl;
        model = new LinearModel(features_number, use_offset);
    } else if (strcmp(model_type.c_str(), "fm") == 0) {
        std::cout << "Using FM model with " << factors_size << " factors." << std::endl;
        model = new FMModel(features_number, factors_size, use_offset);
    } else {
    }
    return model;
}


int main(int argc, char** argv) {
    std::string train_file("../../datasets/train_data.vw");  
    std::string test_file("../../datasets/test_data.vw");  
    std::string model_type("linear");
    size_t factors_size = 10;
    double learning_rate = 1e-3;
    size_t num_epochs = 10;
    bool use_offset = false;
    parse_arguments(argc, argv, &train_file, &test_file, &model_type, &factors_size, &use_offset, &num_epochs, &learning_rate);
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

    Model* model = create_model(model_type, data_reader._features_number, factors_size, use_offset);
    MSE loss;
    std::cout << "Passes number: " << num_epochs << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << std::endl;

    std::cout << "Start to train model..." << std::endl;
    start = clock();
    Optimizer optimizer(num_epochs, learning_rate);
    optimizer.train(model, &loss, x_train, y_train);
    Y train_prediction = model->predict(x_train);
    double train_mse = loss.compute_loss(train_prediction, y_train); 
    finish = clock();
    std::cout << "Training finished! Elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "MSE: " << train_mse << std::endl;
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
    double test_mse = loss.compute_loss(test_prediction, y_test);
    finish = clock();
    std::cout << "Prediction finished! Elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "MSE: " << test_mse << std::endl;

    delete model;
    return 0;
}
