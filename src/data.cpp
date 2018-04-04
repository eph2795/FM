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


DataReader::DataReader(const std::string& filename, bool has_header, size_t target_col)
        : _filename(filename), _has_header(has_header), _target_col(target_col) 
{}


void DataReader::get_columns_info() {
    std::ifstream input(_filename.c_str());
    std::string line;
    for (size_t line_num = 0; std::getline(input, line); line_num++) {
        if ((line_num == 0) && _has_header) {
            continue;
        }
        size_t line_pos = 0;
        for (size_t token_num = 0; line_pos < line.size(); token_num++) {
            std::string token = get_token(&line_pos, line);
            if (token_num == _target_col) {
                continue;
            }
            if (_dtypes.size() == token_num) {
                _dtypes.push_back('I');
                _unique_values.push_back(std::set<std::string>()); 
            };
            _unique_values[token_num].insert(token);
            _dtypes[token_num] = get_token_dtype(_dtypes[token_num], token, token_num);
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