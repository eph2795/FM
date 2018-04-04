#ifndef DATA_H
#define DATA_H

#include <string>
#include <vector>
#include <set>


struct E {
    double value;
    size_t idx;
};


struct Row {
    std::vector<E> elements;
};


struct Matrix {
    std::vector<Row> rows;
};


struct Target {
    std::vector<double> y;
};


struct DataReader {
    std::vector<char> dtypes;
    std::vector<std::set<std::string>> unique_values;
    std::vector<size_t> positions;
};


struct Model {
    std::vector<double> w;
};


std::string get_token(size_t* pos, const std::string& line);


char get_token_dtype(char cur_dtype, const std::string& token, size_t col_number);


void get_columns_info(const std::string& filename, bool has_header, size_t target_col, std::vector<char>* dtypes, std::vector<std::set<std::string>>* unique_values);


void fill_data(const std::string& filename, bool has_header, size_t target_col, const DataReader& data_reader, Matrix* mat, Target* y);

#endif 