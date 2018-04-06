#include <iostream>
#include <string.h>

#include <string>
#include <vector>

#include "data.h"
#include "optimization.h"


void parse_arguments(int argc, char** argv, 
        std::string* train_file, std::string* test_file, size_t* num_epochs, double* learning_rate) {
    for (size_t i = 1; i < argc; i++) {
        try {
            if (strcmp(argv[i], "--data") == 0) {
                i += 1;
                *train_file = std::string(argv[i]); 
            } else if (strcmp(argv[i], "--test") == 0) {
                i += 1;
                *test_file = std::string(argv[i]);
            }else if (strcmp(argv[i], "--passes") == 0) {
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


int main(int argc, char** argv) {
    std::string train_file("../../datasets/train_data.vw");  
    std::string test_file("../../datasets/test_data.vw");  
    double learning_rate = 1e-3;
    size_t num_epochs = 10;
    parse_arguments(argc, argv, &train_file, &test_file, &num_epochs, &learning_rate);

    std::cout << "Passes number: " << num_epochs << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << "Train data file: " << train_file << std::endl;
    std::cout << "Test data file: " << test_file << std::endl;

    DataReader data_reader;
    std::cout << "Start to preprocessing file.." << std::endl;
    data_reader.get_columns_info(train_file);
    std::cout << "File preprocessed!" << std::endl;
 
    std::cout << "Start to reading input data.." << std::endl;
    X x_train;
    Y y_train;  
    data_reader.fill_with_data(train_file, &x_train, &y_train);
    std::cout << "Reading finished!" << std::endl;

    Model model;
    model._w.resize(data_reader._features_number);
    
    
    Optimizer optimizer(num_epochs, learning_rate);

    optimizer.train(&model, x_train, y_train);

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

    Y prediction = model.predict(model, x_test);
    std::cout << "MSE: " << MSE(prediction, y_test) << std::endl;
    return 0;
}
