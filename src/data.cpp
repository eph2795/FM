#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <utility>

#include <boost/spirit/include/qi.hpp>

#include "data.h"


X X::to_csr() const {
    X x_csr;
    x_csr._features_number = _features_number;
    x_csr._objects_number = _objects_number;
    x_csr._data_type = "csr";

    x_csr._objects.resize(x_csr._objects_number);
    size_t object_idx;
    double feature_value;

    for (size_t feature_idx = 0; feature_idx < _features_number; feature_idx++) {
        for (const auto& item: _objects[feature_idx]._items) {
            object_idx = item.first;
            feature_value = item.second;
            x_csr._objects[object_idx]._items.push_back(std::pair<size_t, double>(feature_idx, feature_value));
        }
    } 
    return x_csr;
}


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


DataReader::~DataReader() {}


void DataReader::fill_with_data(const std::string& file_name, X* x, Y* y, const std::string& data_type) const {
    if (strcmp(data_type.c_str(), "csr") == 0) {
        _fill_csr_data(file_name, x, y);
    } else if (strcmp(data_type.c_str(), "csc") == 0) {
        _fill_csc_data(file_name, x, y);
    } else if ((strcmp(data_type.c_str(), "csr") != 0) and ((strcmp(data_type.c_str(), "csc") != 0))) {
        std::cout << "Wrong sparse matrix type! Terminated." << std::endl;
        throw;        
    }
}


void DataReader::_fill_csr_data(const std::string& file_name, X* x, Y* y) const { 
    x->_data_type = std::string("csr");
    x->_features_number = get_features_number();

    std::ifstream input(file_name.c_str());
    std::string line, token;
    for (size_t line_num = 0; std::getline(input, line); line_num++) {
        size_t line_pos = 0;
        
        token = get_token(&line_pos, line, ' ');
        y->_targets.push_back(parse_double(token));

        token = get_token(&line_pos, line, ' ');
        SparseVector object;
        for (size_t token_num = 0; line_pos < line.size(); token_num++) {        
            token = get_token(&line_pos, line, ':');
            size_t idx = this->get_feature_index(token);
            token = get_token(&line_pos, line, ' ');
            double value = parse_double(token);
            object._items.push_back(std::pair<size_t, double>(idx, value)); 
        }
        x->_objects.push_back(object);
    }
    x->_objects_number = y->_targets.size();
}


void DataReader::_fill_csc_data(const std::string& file_name, X* x, Y* y) const { 
    x->_data_type = std::string("csc");
    x->_features_number = get_features_number(); 
    x->_objects.resize(x->_features_number);

    std::ifstream input(file_name.c_str());
    std::string line, token;
    for (size_t line_num = 0; std::getline(input, line); line_num++) {
        size_t line_pos = 0;
        
        token = get_token(&line_pos, line, ' ');
        y->_targets.push_back(parse_double(token));
        
        token = get_token(&line_pos, line, ' ');
        for (size_t token_num = 0; line_pos < line.size(); token_num++) {        
            token = get_token(&line_pos, line, ':');
            size_t idx = this->get_feature_index(token);
            token = get_token(&line_pos, line, ' ');
            double value = parse_double(token);
            
            x->_objects[idx]._items.push_back(std::pair<size_t, double>(line_num, value));
        }
    }
    x->_objects_number = y->_targets.size();
}


void DataReaderOHE::get_columns_info(const std::string& file_name) {
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
}


size_t DataReaderOHE::get_features_number() const {
    return _features_number;
}


size_t DataReaderOHE::get_feature_index(const std::string& feature_name) const {
    return _features_position.at(feature_name);
}


DataReaderHash::DataReaderHash(size_t bits_number): _bits_number(bits_number) 
{}


void DataReaderHash::get_columns_info(const std::string& file_name) {}


size_t DataReaderHash::get_features_number() const {
    return std::pow(2, _bits_number);
}


size_t DataReaderHash::get_feature_index(const std::string& feature_name) const {
    return _hash(feature_name) % this->get_features_number();
}