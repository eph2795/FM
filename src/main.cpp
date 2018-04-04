#include <iostream>

#include <cmath>
#include <algorithm>

#include <string>
#include <set>
#include <vector>

#include "data.h"


double scalar_product(const Row& row, const std::vector<double>& w) {
    double result = 0;
    for (E item: row.elements) {
        result += w[item.idx] * item.value;
    }
    return result;
}

double MSE_grad(const Model& model, const Matrix& mat, const Target& y, size_t row_idx) {
    double result = 0;
    Row row = mat.rows[row_idx];
    for (size_t j = 0; j < row.elements.size(); j++) {
        result += row.elements[j].value * model.w[row.elements[j].idx];
    }
    result = 2 * (result - y.y[row_idx]);
    return result;
}


std::vector<double> model_grad(const Model& model, const Matrix& mat, size_t row_idx) {
    std::vector<double> result(model.w.size(), 0);
    for (size_t i = 0; i < mat.rows[row_idx].elements.size(); i++) {
        E item = mat.rows[row_idx].elements[i];
        result[item.idx] = item.value;
    }
    return result;
}


template<typename T>
void print_vector(const std::vector<T>& v) {
    std::cout << "Vector:" << std::endl;
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl << std::endl;
}


size_t batch_size = 16;
double learning_rate = 1e-1;
size_t num_epochs = 100;
void learn_model(Model* model, const Matrix& mat, const Target& y) {
    size_t N = mat.rows.size();
    std::vector<size_t> rows_order(N);
    std::iota(rows_order.begin(), rows_order.end(), 0);

    for (size_t i = 0; i < num_epochs; i++) {
        std::random_shuffle(rows_order.begin(), rows_order.end());

        std::cout << "Data size: " << N << ", number of batches: " << N / batch_size + (N % batch_size != 0) << std::endl;
        for (size_t j = 0; j < N / batch_size + (N % batch_size != 0); j++) {
            std::vector<size_t> batch_idxes(rows_order.begin() + j * batch_size, rows_order.begin() + std::min((j + 1) * batch_size, N));
            std::vector<double> update(model->w.size(), 0);

            for (size_t row_idx: batch_idxes) {
                double coef = -learning_rate * MSE_grad(*model, mat, y, row_idx);
                std::cout << "Coef for batch " << row_idx << ": " << coef << std::endl;
                std::vector<double> grad = model_grad(*model, mat, row_idx);
                for (size_t k = 0; k < grad.size(); k++) {
                    update[k] += coef * grad[k];
                } 
            }
            
            print_vector(update);
        
            for (size_t k = 0; k < update.size(); k++) {
                model->w[k] += update[k] / batch_size;
            }

            print_vector(model->w);

            // if (j > 3) {
            //     break;
            // }
        }
        print_vector(model->w);
    }
}


std::vector<double> predict(const Model& model, const Matrix& mat) {
    std::vector<double> result;
    for (const Row& row: mat.rows) {
        result.push_back(scalar_product(row, model.w));
    }
    return result;
}


double MSE(const std::vector<double> one, const std::vector<double> another) {
    double result = 0;
    for (size_t i = 0; i < one.size(); i++) {
        std::cout << "Prediction: " << one[i] << ", Target: " << another[i] << std::endl; 
        result += std::pow(one[i] - another[i], 2);
    }
    result /= one.size();
    return result;
}


int main() {
    std::string filename("../../datasets/machine.csv");
    bool has_header = true;
    size_t target_col = 8;
    
    DataReader data_reader;
    std::cout << "Start to preprocessing columns.." << std::endl;
    get_columns_info(filename, has_header, target_col, &(data_reader.dtypes), &(data_reader.unique_values));
 
    data_reader.positions.push_back(0);
    for (size_t i = 0; i < data_reader.unique_values.size(); i++) {
        std::cout << "Dtype: " << data_reader.dtypes[i] << ", unique values: " << data_reader.unique_values[i].size() << "\t";
        if (data_reader.dtypes[i] == 'F') {
            data_reader.positions.push_back(data_reader.positions[i] + 1); 
        } else {
            data_reader.positions.push_back(data_reader.positions[i] + data_reader.unique_values[i].size()); 
        }
    }
    std::cout << std::endl;
    std::cout << "Columns preprocessed!" << std::endl;
 
    std::cout << "Start to filling matrix.." << std::endl;
    Matrix mat;
    Target y;  
    fill_data(filename, has_header, target_col, data_reader, &mat, &y);
    std::cout << "Filling finished!" << std::endl;

    // for (size_t i = 0; i < mat.rows.size(); i++) {
    //     for (size_t j = 0; j < mat.rows[i].elements.size(); j++) {
    //         std::cout << "Idx: " << mat.rows[i].elements[j].idx << ", value: " << mat.rows[i].elements[j].value << "\t";
    //     }
    //     std::cout << std::endl;
    // }

    Model model;
    model.w.resize(data_reader.positions[data_reader.positions.size() - 1]);
    
    learn_model(&model, mat, y);

    print_vector(model.w);

    std::vector<double> prediction = predict(model, mat);
    std::cout << "MSE: " << MSE(prediction, y.y) << std::endl;
    return 0;
}
