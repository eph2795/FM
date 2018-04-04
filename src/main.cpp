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
    std::string filename("../../datasets/machine.csv");
    bool has_header = true;
    size_t target_col = 8;
    
    DataReader data_reader(filename, has_header, target_col);
    std::cout << "Start to preprocessing columns.." << std::endl;
    data_reader.get_columns_info();
    std::cout << "Columns preprocessed!" << std::endl;
 
    std::cout << "Start to filling matrix.." << std::endl;
    X x;
    Y y;  
    data_reader.fill_with_data(&x, &y);
    std::cout << "Filling finished!" << std::endl;

    // for (size_t i = 0; i < mat.rows.size(); i++) {
    //     for (size_t j = 0; j < mat.rows[i].elements.size(); j++) {
    //         std::cout << "Idx: " << mat.rows[i].elements[j].idx << ", value: " << mat.rows[i].elements[j].value << "\t";
    //     }
    //     std::cout << std::endl;
    // }

    Model model;
    model._w.resize(data_reader._positions[data_reader._positions.size() - 1]);
    
    
    size_t batch_size = 16;
    double learning_rate = 1e-1;
    size_t num_epochs = 100;
    Optimizer optimizer(num_epochs, learning_rate, batch_size);

    optimizer.train(&model, x, y);

    print_vector(model._w);

    Y prediction = model.predict(model, x);
    std::cout << "MSE: " << MSE(prediction, y) << std::endl;
    return 0;
}
