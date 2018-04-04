#include <iostream>
#include <fstream>

#include <cassert>

#include <vector>
#include <set>

#include "data.h"


std::string get_token(size_t* pos, const std::string& line, char sep) {
    size_t i;
    for (i = *pos; i < line.size(); i++) {
        if (line[i] == sep) {
            break;
        }
    }
    std::string token = line.substr(*pos, i - *pos);
    *pos = i + 1;
    return token;
}


char get_token_dtype(char cur_dtype, const std::string& token, size_t col_number) {
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


DataReader::DataReader(const std::string& filename): _filename(filename) {}


void DataReader::get_columns_info() {
    std::ifstream input(_filename.c_str());
    std::string line;
    for (size_t line_num = 0; std::getline(input, line); line_num++) {
        size_t line_pos = 0;

        std::string token = get_token(&line_pos, line, ' ');
        try {
            std::stod(token);
        }
        catch {
            std::cout << "Label has wrong type!" << std::endl;
            assert(false);
        }

        token = get_token(&line_pos, line, ' ');
        assert(token.compare("|") != 0);

        for (size_t token_num = 0; line_pos < line.size(); token_num++) {
            std::string token = get_token(&line_pos, line);
            _unique_features.insert(token);
        }
    }

    _positions.push_back(0);
    for (size_t col_num = 0; col_num < _unique_values.size(); col_num++) {
        std::cout << "Dtype: " << _dtypes[col_num] << ", unique values: " << _unique_values[col_num].size() << "\t";
        if (_dtypes[col_num] == 'F') {
            _positions.push_back(_positions[col_num] + 1); 
        } else {
            _positions.push_back(_positions[col_num] + _unique_values[col_num].size()); 
        }
    }
    std::cout << std::endl;
}


void DataReader::fill_with_data(X* x, Y* y) {
    std::ifstream input(_filename.c_str());
    std::string line;
    for (size_t line_num = 0; std::getline(input, line); line_num++) {
        if ((line_num == 0) && _has_header) {
            continue;
        }
        Object object;
        size_t line_pos = 0;
        for (size_t token_num = 0; line_pos < line.size(); token_num++) {
            std::string token = get_token(&line_pos, line);
            if (token_num == _target_col) {
                y->_targets.push_back(std::stod(token));
            } else {
                Feature feauture;
                if (_dtypes[token_num] == 'F') { 
                    feauture.value = std::stod(token);
                    feauture.idx = _positions[token_num];
                } else {
                    feauture.value = 1;
                    feauture.idx = _positions[token_num] + std::distance(_unique_values[token_num].begin(), 
                                                                         _unique_values[token_num].lower_bound(token));  
                }   
                object._features.push_back(feauture);
            }
        }
        x->_objects.push_back(object);
    }
}