#ifndef DATA_H
#define DATA_H

#include <string>
#include <vector>
#include <set>


struct Feature {
    double value;
    size_t idx;
};


struct Object {
    std::vector<Feature> _features;
};


struct X {
    std::vector<Object> _objects;
};


struct Y {
    std::vector<double> _targets;
};


struct DataReader {
    DataReader(const std::string& filename);
    void get_columns_info();
    void fill_with_data(X* x, Y* y);

    std::string _filename;
    size_t _lines_number;
    size_t _features_number;
    std::vector<std::string> _unique_features;
    std::vector<size_t> _features_position;
};


std::string get_token(size_t* pos, const std::string& line);


char get_token_dtype(char cur_dtype, const std::string& token, size_t col_number);


#endif 