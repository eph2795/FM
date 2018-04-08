#ifndef DATA_H
#define DATA_H

#include <string>
#include <vector>
#include <map>


struct SparseVector {
    std::map<size_t, double> _items;
};


struct X {
    std::vector<SparseVector> _objects;
};


struct Y {
    std::vector<double> _targets;
};


struct DataReader {
    DataReader() {};
    void get_columns_info(const std::string& filename);
    void fill_with_data(const std::string& filename, X* x, Y* y) const ;

    size_t _features_number;
    size_t _objects_number;
    std::map<std::string, size_t> _features_position;
};


std::string get_token(size_t* pos, const std::string& line, char sep);


#endif 