#include <iostream>
#include <fstream>

#include <vector>
#include <set>

#include "data.h"


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


void get_columns_info(const std::string& filename, bool has_header, size_t target_col, 
                      std::vector<char>* dtypes, std::vector<std::set<std::string>>* unique_values) {
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
            if (i == target_col) {
                continue;
            }
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

void fill_data(const std::string& filename, bool has_header, size_t target_col, 
               const DataReader& data_reader, Matrix* mat, Target* y) {
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
            if (i == target_col) {
                y->y.push_back(std::stod(token));
            } else {
                E cur;
                if (data_reader.dtypes[i] == 'F') { 
                    cur.value = std::stod(token);
                    cur.idx = data_reader.positions[i];
                } else {
                    cur.value = 1;
                    cur.idx = data_reader.positions[i] 
                        + std::distance(data_reader.unique_values[i].begin(), 
                                        data_reader.unique_values[i].lower_bound(token));  
                }   
                row.elements.push_back(cur);
            }
        }
        mat->rows.push_back(row);
    }
}