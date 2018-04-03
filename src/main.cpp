#include <iostream>
#include <fstream>

#include <cmath>
#include <algorithm>

#include <string>
#include <set>
#include <vector>


struct E {
    double value;
    size_t idx;
};


struct Row {
    std::vector<E> elements;
};


struct Matrix {
    std::vector<char> dtypes;
    std::vector<std::set<std::string>> unique_values;
    std::vector<size_t> positions;
    std::vector<Row> rows;
    std::vector<double> w;
    size_t target;
};


std::set<char> seps = {' ', '\n', '\t', ','};

std::string get_token(size_t* pos, const std::string& line) {
    size_t i;
    for (i = *pos; i < line.size(); i++) {
        if (seps.find(line[i]) != seps.end()) {
            break;
        }
    }
    std::string token = line.substr(*pos, i - *pos);
    *pos = i + 1;
    return token;
}


char get_token_dtype(char cur_dtype, const std::string& token, size_t col_number) {
    size_t pos = 0;
    if (cur_dtype == 'I') {
        try {
            std::stoi(token, &pos);
            if ((cur_dtype == 'I') && (pos == token.size())) {
                return 'I';
            }
        } 
        catch (std::invalid_argument e) {
            std::cout << "This column is not Int: " << col_number << ", example: " << token << "!" << std::endl;
        } 
    }      
    if (cur_dtype != 'C') {
        try {
            std::stod(token, &pos);
            if ((cur_dtype != 'C') && (pos == token.size())) {
                return 'F';
            } 
        }
        catch (std::invalid_argument e) {
            std::cout << "This column is not Float: " << col_number << ", example: " << token << "!" << std::endl;
        }
    }
    return 'C';
}


void get_columns_info(const std::string& filename, bool has_header, std::vector<char>* dtypes, std::vector<std::set<std::string>>* unique_values) {
    std::ifstream input(filename.c_str());
    std::string line;

    // std::cout << "Start to processing info..." << std::endl;
    for (size_t k = 0; std::getline(input, line); k++) {
        if ((k == 0) && has_header) {
            continue;
        }
        size_t pos = 0;
        for (size_t i = 0; pos < line.size(); i++) {
            std::string token = get_token(&pos, line);
            if (dtypes->size() == i) {
                dtypes->push_back('I');
                unique_values->push_back(std::set<std::string>()); 
            };
            (*unique_values)[i].insert(token);
            (*dtypes)[i] = get_token_dtype((*dtypes)[i], token, i);
        }
        // std::cout << "Another line handled.." << std::endl;
    }

    // std::cout << "Info processing finished!" << std::endl;
    // for (size_t i = 0; i < dtypes->size(); i++) {
    //     num_values->push_back(unique_values[i].size());
    //     // std::cout << dtypes[i] << " - " << unique_values[i].size() << "\t";
    // }
    // std::cout << "\n\n";
}

void fill_data(Matrix& mat, const std::string& filename, bool has_header) {
    std::ifstream input(filename.c_str());
    std::string line;
    for (size_t k = 0; std::getline(input, line); k++) {
        if ((k == 0) && has_header) {
            continue;
        }
        Row row;
        size_t pos = 0;
        for (size_t i = 0; pos < line.size(); i++) {
            std::string token = get_token(&pos, line);
            E cur;
            if (mat.dtypes[i] == 'F') { 
                cur.value = std::stod(token);
                cur.idx = mat.positions[i];
            } else {
                cur.value = 1;
                cur.idx = mat.positions[i] + std::distance(mat.unique_values[i].begin(), mat.unique_values[i].lower_bound(token));  
            }   
            row.elements.push_back(cur);
        }
        mat.rows.push_back(row);
    }
}


double scalar_product(const Row& row, const std::vector<double>& w) {
    double result = 0;
    for (E item: row.elements) {
        result += w[item.idx] * item.value;
    }
    return result;
}

double MSE_grad(Matrix& mat, size_t idx) {
    double result = 0;
    Row row = mat.rows[idx];
    for (size_t j = 0; j < row.elements.size(); j++) {
        if (j == mat.target) {
            result -= row.elements[j].value;
        } else {
            result += row.elements[j].value * mat.w[row.elements[j].idx];
        }
    }
    result = 2 * result;
    return result;
}


std::vector<double> model_grad(Matrix& mat, size_t idx) {
    std::vector<double> result(mat.w.size(), 0);
    for (size_t i = 0; i < mat.rows[idx].elements.size(); i++) {
        if (i != mat.target) {
            E item = mat.rows[idx].elements[i];
            result[item.idx] = item.value;
        }
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
double learning_rate = 1e-6;
size_t num_epochs = 10;
void learn_model(Matrix& mat) {
    size_t N = mat.rows.size();
    std::vector<size_t> rows_order(N);
    std::iota(rows_order.begin(), rows_order.end(), 0);

    for (size_t i = 0; i < num_epochs; i++) {
        std::random_shuffle(rows_order.begin(), rows_order.end());

        std::cout << "Data size: " << N << ", number of batches: " << N / batch_size + (N % batch_size != 0) << std::endl;
        for (size_t j = 0; j < N / batch_size + (N % batch_size != 0); j++) {
            std::vector<size_t> batch_idxes(rows_order.begin() + j * batch_size, rows_order.begin() + std::min((j + 1) * batch_size, N));
            std::vector<double> update(mat.w.size(), 0);

            for (size_t idx: batch_idxes) {
                double coef = -learning_rate * MSE_grad(mat, idx);
                std::cout << "Coef for batch " << idx << ": " << coef << std::endl;
                std::vector<double> grad = model_grad(mat, idx);
                for (size_t k = 0; k < grad.size(); k++) {
                    update[k] += coef * grad[k];
                } 
            }
            
            print_vector(update);
        
            for (size_t k = 0; k < update.size(); k++) {
                mat.w[k] += update[k] / batch_size;
            }

            print_vector(mat.w);

            // if (j > 3) {
            //     break;
            // }
        }
        print_vector(mat.w);
    }
}


std::vector<double> predict(const Matrix& mat) {
    std::vector<double> result;
    for (const Row& row: mat.rows) {
        result.push_back(scalar_product(row, mat.w));
    }
    return result;
}


double MSE(const std::vector<double> one, const std::vector<double> another) {
    double result = 0;
    for (size_t i = 0; i < one.size(); i++) {
        result += std::pow(one[i] - another[i], 2);
    }
    result /= one.size();
    return result;
}


int main() {
    std::string filename("../../datasets/forestfires.csv");

    Matrix mat;
    mat.target = 12;

    std::cout << "Start to preprocessing columns.." << std::endl;
    get_columns_info(filename, true, &(mat.dtypes), &(mat.unique_values));
    mat.positions.push_back(0);
    for (size_t i = 0; i < mat.unique_values.size(); i++) {
        std::cout << "Dtype: " << mat.dtypes[i] << ", unique values: " << mat.unique_values[i].size() << "\t";
        if (mat.dtypes[i] == 'F') {
            mat.positions.push_back(mat.positions[i] + 1); 
        } else {
            mat.positions.push_back(mat.positions[i] + mat.unique_values[i].size()); 
        }
    }
    std::cout << std::endl;
    mat.w.resize(mat.positions[mat.positions.size() - 1]);

    std::cout << "Columns preprocessed!" << std::endl;

    std::cout << "Start to filling matrix.." << std::endl;
    fill_data(mat, filename, true);
    std::cout << "Filling finished!" << std::endl;

    // for (size_t i = 0; i < mat.rows.size(); i++) {
    //     for (size_t j = 0; j < mat.rows[i].elements.size(); j++) {
    //         std::cout << "Idx: " << mat.rows[i].elements[j].idx << ", value: " << mat.rows[i].elements[j].value << "\t";
    //     }
    //     std::cout << std::endl;
    // }

    learn_model(mat);

    print_vector(mat.w);

    std::vector<double> prediction = predict(mat);
    std::vector<double> target;
    for (size_t i = 0; i < mat.rows.size(); i++) {
        target.push_back(mat.rows[i].elements[mat.target].value);
    }
    std::cout << "MSE: " << MSE(prediction, target) << std::endl;
    return 0;
}
