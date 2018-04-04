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
    DataReader(const std::string& filename, bool has_header, size_t target_col);
    void get_columns_info();
    void fill_with_data(X* x, Y* y);

    std::string _filename;
    bool _has_header;
    size_t _target_col;

    std::vector<char> _dtypes;
    std::vector<std::set<std::string>> _unique_values;
    std::vector<size_t> _positions;
};


std::string get_token(size_t* pos, const std::string& line);


char get_token_dtype(char cur_dtype, const std::string& token, size_t col_number);


#endif 