#include <iostream>
#include <fstream>
#include <string.h>
#include <string>
#include <vector>
#include <ctime>

#include "data.h"
#include "model.h"
#include "loss.h"
#include "optimization.h"
#include "regularizer.h"        


void parse_arguments(int argc, char** argv, 
        std::string* train_file, std::string* test_file, std::string* model_type, std::string* loss_type, 
        size_t* factors_size, bool* use_offset, size_t* num_epochs, double* learning_rate,
        std::string* reg_type, double* C, bool* use_common, double* C0, double* Cv, double* Cw, bool* adaptive_reg,
        std::string* index_type, size_t* bits_number, std::string* validation_file, 
        std::string* optimizer_type, std::string* model_file, std::string* predict_file, 
        bool* use_train, bool* use_validation, bool* use_test, bool* dump, bool* load, bool* predict) {
    for (size_t i = 1; i < static_cast<size_t>(argc); i++) {
        try {
            if (strcmp(argv[i], "--train") == 0) {
                i += 1;
                *use_train = true;
                *train_file = std::string(argv[i]); 
            } else if (strcmp(argv[i], "--validation") == 0) {
                i += 1;
                *use_validation = true;
                *validation_file = std::string(argv[i]);
            } else if (strcmp(argv[i], "--test") == 0) {
                i += 1;
                *use_test = true;
                *test_file = std::string(argv[i]);
            } else if (strcmp(argv[i], "--model") == 0) {
                i += 1;
                *model_type = std::string(argv[i]);
            } else if (strcmp(argv[i], "--predict") == 0) {
                i += 1;
                *predict = true;
                *predict_file = std::string(argv[i]);
            } else if (strcmp(argv[i], "--loss") == 0) {
                i += 1;
                *loss_type = std::string(argv[i]);
            } else if (strcmp(argv[i], "--factors_size") == 0) {
                i += 1;
                *factors_size = std::stoul(argv[i]);
            } else if (strcmp(argv[i], "--use_offset") == 0) {
                *use_offset = true;
            } else if (strcmp(argv[i], "--passes") == 0) {
                i += 1;
                *num_epochs = std::stoul(argv[i]);
            } else if (strcmp(argv[i], "--learning_rate") == 0) {
                i += 1;
                *learning_rate = std::stod(argv[i]);
            } else if (strcmp(argv[i], "--reg_type") == 0) {
                i += 1;
                *reg_type = std::string(argv[i]);           
            } else if (strcmp(argv[i], "-C") == 0) {
                i += 1;
                *use_common = true;
                *C = std::stod(argv[i]);
            } else if (strcmp(argv[i], "-C0") == 0) {
                i += 1;
                *C0 = std::stod(argv[i]);
            } else if (strcmp(argv[i], "-Cw") == 0) {
                i += 1;
                *Cw = std::stod(argv[i]);
            } else if (strcmp(argv[i], "-Cv") == 0) {
                i += 1;
                *Cv = std::stod(argv[i]);
            } else if (strcmp(argv[i], "--adaptive_reg") == 0) {
                *adaptive_reg = true;
            } else if (strcmp(argv[i], "--index_type") == 0) {
                i += 1;
                *index_type = std::string(argv[i]);
            } else if (strcmp(argv[i], "--bits_number") == 0) {
                i += 1;
                *bits_number = std::stoul(argv[i]);
            } else if (strcmp(argv[i], "--optimizer") == 0) {
                i += 1;
                *optimizer_type = std::string(argv[i]);
            } else if (strcmp(argv[i], "--dump") == 0) {
                i += 1;
                *dump = true;
                *model_file = std::string(argv[i]);
            } else if (strcmp(argv[i], "--load") == 0) {
                i += 1;
                *load = true;
                *model_file = std::string(argv[i]);
            }
        } 
        catch (std::invalid_argument) {
            throw std::invalid_argument("Wrong arguments format!");
        }
    }
}


Regularizer* create_regularizer(const std::string& reg_type) {
    Regularizer* regularizer;
    if (strcmp(reg_type.c_str(), "l1") == 0) {
        std::cout << "Reqularizer: L1." << std::endl;
        regularizer = new L1();
    } else if (strcmp(reg_type.c_str(), "l2") == 0) {
        std::cout << "Regularizer: L2." << std::endl;
        regularizer = new L2();
    } else {
        std::cout << "Wrong regularizer name! Terminated." << std::endl;
        throw;
    }
    std::cout << std::endl;
    return regularizer;
}


Model* create_model(const std::string& model_type, size_t features_number, size_t factors_size, bool use_offset,
                    const std::string& reg_type, double C0, double Cw, double Cv) {
    Regularizer* regularizer = create_regularizer(reg_type);
    Model* model;
    if (use_offset) {
        std::cout << "With offset w0." << std::endl;
    } else {
        std::cout << "Without offset w0." << std::endl;
    }
    if (strcmp(model_type.c_str(), "linear") == 0) {
        std::cout << "Model: linear." << std::endl;
        std::cout << "\tC0=" << C0 << ", Cw=" << Cw << "." << std::endl;
        model = new LinearModel(features_number, use_offset, C0, Cw, regularizer);
    } else if (strcmp(model_type.c_str(), "fm") == 0) {
        std::cout << "Model: FM, " << " factors size=" << factors_size << "." << std::endl;
        std::cout << "\tC0=" << C0 << ", Cw=" << Cw << ", Cv=" << Cv << "." << std::endl;
        model = new FMModel(features_number, factors_size, use_offset, C0, Cw, std::vector<double>(factors_size, Cv), regularizer);
    } else {
        std::cout << "Wrong model name! Terminated." << std::endl;
        throw;
    }
    std::cout << std::endl;
    return model;
}


Loss* create_loss(const std::string& loss_type) {
    Loss* loss;
    if (strcmp(loss_type.c_str(), "mse") == 0) {
        std::cout << "Loss: MSE." << std::endl;
        loss = new MSE();
    } else if (strcmp(loss_type.c_str(), "logistic") == 0) {
        std::cout << "Loss: logistic." << std::endl;
        loss = new Logistic();
    } else {
        std::cout << "Wrong loss name! Terminated." << std::endl;
        throw;
    }
    std::cout << std::endl;
    return loss;
}


DataReader* create_reader(const std::string& index_type, size_t bits_number) {
    DataReader* data_reader;
    if (strcmp(index_type.c_str(), "ohe") == 0) {
        std::cout << "Features type: OHE." << std::endl;
        data_reader = new DataReaderOHE();
    } else if (strcmp(index_type.c_str(), "hash") == 0) {
        std::cout << "Features type: hashing, bits number=" << bits_number << "." << std::endl;
        data_reader = new DataReaderHash(bits_number); 
    } else {
        std::cout << "Wrong index type! Terminated." << std::endl;
        throw;
    }
    std::cout << std::endl;
    return data_reader;
}


Optimizer* create_optimizer(const std::string& optimizer_type, size_t num_epochs, double learning_rate) {
    Optimizer* optimizer;
    std::cout << "Passes number: " << num_epochs << std::endl;
    if (strcmp(optimizer_type.c_str(), "sgd")  == 0) {
        std::cout << "Optimizer: SGD." << std::endl;
        optimizer = new SGDOptimizer(num_epochs, learning_rate);
        std::cout << "\tLearning rate: " << learning_rate << std::endl;
    } else if (strcmp(optimizer_type.c_str(), "als") == 0) {
        std::cout << "Optimizer: ALS." << std::endl;
        optimizer = new ALSOptimizer(num_epochs);
    } else {
        std::cout << "Wrong optimizer type! Terminated." << std::endl;
        throw;
    }
    std::cout << std::endl;
    return optimizer;
}


int main(int argc, char** argv) {
    std::string train_file, validation_file, test_file, model_file, predict_file;
    bool use_train = false, use_validation = false, use_test = false, dump = false, load = false, predict=false;

    std::string model_type("linear");
    std::string loss_type("mse");
    std::string reg_type("l2");
    std::string index_type("ohe");
    std::string optimizer_type("sgd");

    size_t bits_number = 10;
    size_t factors_size = 10;
    double learning_rate = 1e-3;
    double C = 1e-3, C0 = 1, Cw = 1e-6, Cv = 1e-3;
    size_t num_epochs = 10;
    bool use_offset = false, use_common = false, adaptive_reg = false;

    parse_arguments(argc, argv, &train_file, &test_file, &model_type, &loss_type, &factors_size, &use_offset, 
        &num_epochs, &learning_rate, 
        &reg_type, &C, &use_common, &C0, &Cw, &Cv, &adaptive_reg,
        &index_type, &bits_number, &validation_file,
        &optimizer_type, &model_file, &predict_file, &use_train, &use_validation, &use_test, &dump, &load, &predict);
    if (use_common) {
        C0 = C;
        Cw = C;
        Cv = C;
    }
    if (use_train and load) {
        std::cout << "Can't retrain loaded model! Terminated." << std::endl;
        throw;
    } else if (not use_train and not load) {
        std::cout << "Specify train or load option! Terminated." << std::endl;
        throw;
    } else if (adaptive_reg and not use_validation) {
        std::cout << "Can't use adaptive regularization without validation set! Terminated." << std::endl;
        throw;
    }

    clock_t start, finish;

    if (use_train) {
        std::cout << "Train data file: " << train_file << std::endl;
    }
    if (use_validation) {
        std::cout << "Validation data file: " << validation_file << std::endl;
    }
    if (use_test) {
        std::cout << "Test data file: " << test_file << std::endl;
    }
    std::cout << std::endl;

    DataReader* data_reader = create_reader(index_type, bits_number);
    if (strcmp(index_type.c_str(), "ohe") == 0) {
        if (not use_train) {
            std::cout << "You must use train file with ohe features! Terminated." << std::endl;
            throw;
        }
        std::cout << "Start to preprocessing train file..." << std::endl;
        start = clock();
        data_reader->get_columns_info(train_file);
        finish = clock();
        std::cout << "Train file preprocessed! Elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
        std::cout << std::endl;
    }
    

    X x_train, x_val, x_test;
    Y y_train, y_val, y_test;
    if (use_train) {
        std::cout << "Start to reading train data..." << std::endl;
        start = clock();
        data_reader->fill_with_data(train_file, &x_train, &y_train);
        finish = clock();
        std::cout << "Reading finished! Elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
        std::cout << std::endl;
    }
    if (use_validation) {
        std::cout << "Start to reading validation data..." << std::endl;
        start = clock();
        data_reader->fill_with_data(validation_file, &x_val, &y_val);
        finish = clock();
        std::cout << "Reading finished! Elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
        std::cout << std::endl;
    }
    
    
    Model* model = create_model(model_type, data_reader->get_features_number(), factors_size, use_offset, reg_type, C0, Cw, Cv);
    Loss* loss = create_loss(loss_type);
    
    if (use_train) {
        Optimizer* optimizer = create_optimizer(optimizer_type, num_epochs, learning_rate);

        std::cout << "Start to train model..." << std::endl;
        start = clock();
        optimizer->train(model, loss, x_train, y_train, use_validation, x_val, y_val, adaptive_reg);
    
        Y train_prediction = model->predict(x_train);
        double train_mse = loss->compute_loss(train_prediction, y_train); 
        finish = clock();
        std::cout << "Training finished! Elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
        std::cout << "Loss: " << train_mse << std::endl;
        std::cout << std::endl;
        delete optimizer;
    } else if (load) {
        std::cout << "Loading model, path: " << model_file << std::endl;
        model->load(model_file);
        std::cout << "Model loaded." << std::endl;
        std::cout << std::endl;
    } 
    
    if (dump) {
        std::cout << "Start to dump model, path: " << model_file << std::endl;
        model->dump(model_file);
        std::cout << "Model dumped." << std::endl;
        std::cout << std::endl;
    }

    if (use_test) {
        std::cout << "Start to reading test data..." << std::endl;
        start = clock();
        data_reader->fill_with_data(test_file, &x_test, &y_test);
        finish = clock();
        std::cout << "Reading finished! Elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
    
        std::cout << "Start to predict on test data..." << std::endl;
        start = clock();
        Y test_prediction = model->predict(x_test);
        double test_mse = loss->compute_loss(test_prediction, y_test);
        finish = clock();
        std::cout << "Prediction finished! Elapsed time: " << double(finish - start) / CLOCKS_PER_SEC << std::endl;
        std::cout << "Loss: " << test_mse << std::endl;

        if (predict) {
            std::cout << "Dumping prediction in file, path: " << predict_file << std::endl;
            std::ofstream file;
            file.open(predict_file.c_str(), std::ios::out | std::ios::binary);

            if (file.is_open()) {
                for (auto y: test_prediction._targets) {
                    file << y << std::endl;
                }
                file.close();
            } else {
                std::cout << "Unable to open predict file! Terminated." << std::endl;
                throw;
            }
        }
        std::cout << std::endl;
    }

    delete data_reader;
    delete model;
    delete loss;
    return 0;
}
