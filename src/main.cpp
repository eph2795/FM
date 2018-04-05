#include <iostream>

#include <string>
#include <vector>

#include "data.h"
#include "optimization.h"


template<typename T>
void print_vector(const std::vector<T>& v) {
    std::cout << "Vector:" << std::endl;
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl << std::endl;
}


int main() {
    std::string filename("../../datasets/u.data.vw");    
    DataReader data_reader(filename);
    std::cout << "Start to preprocessing columns.." << std::endl;
    data_reader.get_columns_info();
    std::cout << "Columns preprocessed!" << std::endl;
 
    std::cout << "Start to filling matrix.." << std::endl;
    X x;
    Y y;  
    data_reader.fill_with_data(&x, &y);
    std::cout << "Filling finished!" << std::endl;

    // for (size_t i = 0; i < x._objects.size(); i++) {
    //     for (size_t j = 0; j < x._objects[i]._features.size(); j++) {
    //         std::cout << "Idx: " << x._objects[i]._features[j].idx << ", value: " << x._objects[i]._features[j].value << "\t";
    //     }
    //     std::cout << "\t target: "  << y._targets[i] << std::endl;
    // }

    Model model;
    model._w.resize(data_reader._features_number);
    
    
    size_t batch_size = 16;
    double learning_rate = 1e-1;
    size_t num_epochs = 10;
    Optimizer optimizer(num_epochs, learning_rate, batch_size);

    optimizer.train(&model, x, y);

    // print_vector(model._w);

    Y prediction = model.predict(model, x);
    std::cout << "MSE: " << MSE(prediction, y) << std::endl;
    return 0;
}
