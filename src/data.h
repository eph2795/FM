#ifndef DATA_H
#define DATA_H


#include <utility>
#include <string>
#include <vector>
#include <unordered_map>


struct SparseVector {
    std::vector<std::pair<size_t, double>> _items;
};


struct X {
    std::vector<SparseVector> _objects;
};


struct Y {
    std::vector<double> _targets;
};


struct DataReader {
    virtual ~DataReader() = 0;

    virtual void get_columns_info(const std::string& filename) = 0;
    virtual void fill_with_data(const std::string& filename, X* x, Y* y) const = 0;

    size_t _features_number;
};


struct DataReaderOHE: DataReader {
    DataReaderOHE() {};

    void get_columns_info(const std::string& filename);
    void fill_with_data(const std::string& filename, X* x, Y* y) const ;

    size_t _features_number;
    size_t _objects_number;
    std::unordered_map<std::string, size_t> _features_position;
};


struct DataReaderHash: DataReader {
    DataReaderHash(size_t bits_number);

    void get_columns_info(const std::string& filename);
    void fill_with_data(const std::string& filename, X* x, Y* y) const ;

    size_t _features_number;
    size_t _bits_number;
    std::hash<std::string> _hash;
};


std::string get_token(size_t* pos, const std::string& line, char sep);


#endif 