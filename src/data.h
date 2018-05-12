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
    std::string _data_type;
    size_t _features_number;
    size_t _objects_number;
};


struct Y {
    std::vector<double> _targets;
};


struct DataReader {
    virtual ~DataReader() = 0;

    virtual void get_columns_info(const std::string& filename) = 0;
    virtual size_t get_features_number() const = 0;
    virtual size_t get_feature_index(const std::string& feature_name) const = 0;
    void fill_with_data(const std::string& filename, X* x, Y* y, const std::string& data_type) const;
    void _fill_csr_data(const std::string& filename, X* x, Y* y) const;
    void _fill_csc_data(const std::string& filename, X* x, Y* y) const;
};


struct DataReaderOHE: DataReader {
    DataReaderOHE() {};

    void get_columns_info(const std::string& filename);
    size_t get_features_number() const;
    size_t get_feature_index(const std::string& feature_name) const;
    // void fill_with_data(const std::string& filename, X* x, Y* y) const ;

    size_t _features_number;
    size_t _objects_number;
    std::unordered_map<std::string, size_t> _features_position;
};


struct DataReaderHash: DataReader {
    DataReaderHash(size_t bits_number);

    void get_columns_info(const std::string& filename);
    size_t get_features_number() const;
    size_t get_feature_index(const std::string& feature_name) const;
    // void fill_with_data(const std::string& filename, X* x, Y* y) const ;

    size_t _bits_number;
    std::hash<std::string> _hash;
};


std::string get_token(size_t* pos, const std::string& line, char sep);


#endif 