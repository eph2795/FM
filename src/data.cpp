#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <utility>

#include <boost/spirit/include/qi.hpp>

#include "data.h"


std::string get_token(size_t* pos, const std::string& line, char sep) {
    size_t i = line.find(sep, *pos);
    if (i == std::string::npos) {
        i = line.size();
    }
    std::string token = line.substr(*pos, i - *pos);
    *pos = i + 1;
    return token;
}


double parse_double(const std::string& token) {
    const char* ptr = token.c_str();
    size_t size = token.size();
    double result;
    boost::spirit::qi::parse(ptr, ptr + size, boost::spirit::qi::double_, result);
    return result;
}


void DataReader::get_columns_info(const std::string& file_name) {
    _features_number = 0;

    std::ifstream input(file_name.c_str());
    std::string line, token;
    size_t line_num;
    for (line_num = 0; std::getline(input, line); line_num++) {
        size_t line_pos = 0;

        token = get_token(&line_pos, line, ' ');
        try {
            parse_double(token);
            // std::stod(token);
        }
        catch (std::invalid_argument) {
            std::cout << "Label has wrong type!" << std::endl;
            assert(false);
        }
        token = get_token(&line_pos, line, ' ');
        assert(token[0] == '|');

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
    std::cout << "Number of objects: " << _objects_number << ", Number of features: " << _features_number << std::endl;
    // for (std::pair<std::string, size_t> item: _features_position) {
    //     std::cout << "Feature: " << item.first << ", column number: " << item.second << "\t";
    // }
    // std::cout << std::endl;
}


void DataReader::fill_with_data(const std::string& file_name, X* x, Y* y) const {
    // x->_objects.resize(_objects_number);
    // y->_targets.resize(_objects_number);

    std::ifstream input(file_name.c_str());
    std::string line, token;
    for (size_t line_num = 0; std::getline(input, line); line_num++) {
        size_t line_pos = 0;
        
        token = get_token(&line_pos, line, ' ');
        y->_targets.push_back(parse_double(token));
        // y->_targets.push_back(std::stod(token));

        token = get_token(&line_pos, line, ' ');
        SparseVector object;
        for (size_t token_num = 0; line_pos < line.size(); token_num++) {        
            token = get_token(&line_pos, line, ':');
            size_t idx = _features_position.at(token);
            token = get_token(&line_pos, line, ' ');
            double value = parse_double(token);
            // feauture.value = std::stod(token);
            // object._items[idx] = value;
            object._items.push_back(std::pair<size_t, double>(idx, value));
        }
        x->_objects.push_back(object);
    }
}