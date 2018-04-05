#include <iostream>
#include <fstream>

#include <cassert>

#include <vector>
#include <utility>
#include <map>

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


// char get_token_dtype(char cur_dtype, const std::string& token, size_t col_number) {
//     if (cur_dtype == 'I') {
//         try {
//             std::stoi(token, &pos);
//             if ((cur_dtype == 'I') && (pos == token.size())) {
//                 return 'I';
//             }
//         } 
//         catch (std::invalid_argument e) {
//             std::cout << "This column is not Int: " << col_number << ", example: " << token << "!" << std::endl;
//         } 
//     }      
//     if (cur_dtype != 'C') {
//         try {
//             std::stod(token, &pos);
//             if ((cur_dtype != 'C') && (pos == token.size())) {
//                 return 'F';
//             } 
//         }
//         catch (std::invalid_argument e) {
//             std::cout << "This column is not Float: " << col_number << ", example: " << token << "!" << std::endl;
//         }
//     }
//     return 'C';
// }


DataReader::DataReader(const std::string& filename): _filename(filename) {}


void DataReader::get_columns_info() {
    _features_number = 0;

    std::ifstream input(_filename.c_str());
    std::string line, token;
    size_t line_num;
    for (line_num = 0; std::getline(input, line); line_num++) {
        size_t line_pos = 0;

        token = get_token(&line_pos, line, ' ');
        try {
            std::stod(token);
        }
        catch (std::invalid_argument) {
            std::cout << "Label has wrong type!" << std::endl;
            assert(false);
        }
        token = get_token(&line_pos, line, ' ');
        assert(token.compare("|") == 0);

        for (size_t token_num = 0; line_pos < line.size(); token_num++) {
            token = get_token(&line_pos, line, ':');
            if (_features_position.find(token) == _features_position.end()) {
                _features_position[token] = _features_number;
                _features_number += 1;
            }
            token = get_token(&line_pos, line, ' ');
        }
    }

    _objects_number = line_num;
    std::cout << _objects_number << " " << _features_number << std::endl;
    // for (std::pair<std::string, size_t> item: _features_position) {
    //     std::cout << "Feature: " << item.first << ", column number: " << item.second << "\t";
    // }
    // std::cout << std::endl;
}


void DataReader::fill_with_data(X* x, Y* y) {
    x->_objects.resize(_objects_number);
    y->_targets.resize(_objects_number);

    std::ifstream input(_filename.c_str());
    std::string line, token;
    for (size_t line_num = 0; std::getline(input, line); line_num++) {
        size_t line_pos = 0;
        
        token = get_token(&line_pos, line, ' ');
        y->_targets[line_num] = std::stod(token);

        token = get_token(&line_pos, line, ' ');
        Object object;
        for (size_t token_num = 0; line_pos < line.size(); token_num++) {
            Feature feauture;         
            token = get_token(&line_pos, line, ':');
            feauture.idx = _features_position[token];
            token = get_token(&line_pos, line, ' ');
            feauture.value = std::stod(token);
            object._features.push_back(feauture);
        }
        x->_objects[line_num] = object;
    }
}