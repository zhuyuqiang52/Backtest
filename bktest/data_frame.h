#pragma once
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <unordered_set>
#include <boost/functional/hash.hpp>
#include <set>
// T indicates the type of the index column
//default to set index as a increasing vector with type size_t
template <typename T = size_t>
class data_frame {
private:
    Eigen::MatrixXd data;
    std::vector<std::string> column_names;
    std::vector<T> index;

public:
    data_frame() {};
    data_frame(Eigen::MatrixXd data, std::vector<std::string> column_names, std::vector<T> index);
    virtual ~data_frame() {};
    //read data from file
    static data_frame read_csv(std::string file_path, const int index_col = -1);
    static boost::posix_time::ptime string_to_ptime(std::string time_str);
    template <typename U>
    static data_frame<U> set_index(data_frame<T>& lhs, std::vector<U>& index);
    void set_index(std::vector<T> index);
    void assign_index(std::vector<T> new_index);
    Eigen::MatrixXd get_data() const;
    std::vector<std::string> get_column_names();
    std::vector<T> get_index();
    void set_data(Eigen::MatrixXd data);
    void set_column_names(std::vector<std::string> column_names);
    //data info
    int rows() const;
    int cols() const;
    //data selection
    data_frame<T> get_columns(const std::vector<std::string> column_names) const;
    data_frame<T> get_columns(const std::vector<int> column_index) const;
    data_frame<T> get_rows(const std::vector<T> index) const;
    data_frame<T> get_rows(const std::vector<int> index) const;
    data_frame<T> get_rows(T index) const;
    data_frame<T> get_rows(int index) const;
    //data modification
    void insert_row(const int row_index, const T idx, const Eigen::MatrixXd& row);
    data_frame<T> dropna(const int axis = 0, const std::string how = "any");
    double& operator()(const int row_index, const int col_index);
    //data view
    void head(const int n) const;
    void tail(const int n) const;
    //data concat
    data_frame<T> left_join(data_frame<T>& df2) const;
    void row_concat(data_frame<T>& df2);
    //computing function
    data_frame<T> log();
    data_frame<T> exp();
    data_frame<T> row_diff();
};

template <typename T>
int data_frame<T>::rows() const {
    return this->data.rows();
}

template <typename T>
int data_frame<T>::cols() const {
    return this->data.cols();
}

template <typename T>
boost::posix_time::ptime data_frame<T>::string_to_ptime(std::string time_str) {
    boost::posix_time::ptime time = boost::posix_time::time_from_string(time_str + " 00:00:00");
    return time;
}
template <typename T>
data_frame<T>::data_frame(Eigen::MatrixXd data, std::vector<std::string> column_names, std::vector<T> index) {
    this->data = data;
    this->column_names = column_names;
    this->index = index;
}
template <typename T>
void data_frame<T>::assign_index(std::vector<T> new_index) {
    if (new_index.size() != this->data.rows()) {
        std::cerr << "Number of index values does not match number of rows in data" << std::endl;
    }
    this->index = new_index;
}
template <typename T>
Eigen::MatrixXd data_frame<T>::get_data() const {
    Eigen::MatrixXd data_copy(this->data);
    return data_copy;
}
template <typename T>
std::vector<std::string> data_frame<T>::get_column_names() {
    std::vector<std::string> column_names_copy = this->column_names;
    return column_names_copy;
}
template <typename T>
std::vector<T> data_frame<T>::get_index() {
    std::vector<T> index_copy = this->index;
    return index_copy;
}
template <typename T>
void data_frame<T>::set_data(Eigen::MatrixXd data) {
    this->data = data;
}
template <typename T>
void data_frame<T>::set_column_names(std::vector<std::string> column_names) {
    if (column_names.size() != this->data.cols()) {
        std::cerr << "Number of column names does not match number of columns in data" << std::endl;
    }
    this->column_names = column_names;
}
template <typename T>
template <typename U>
data_frame<U> data_frame<T>::set_index(data_frame<T>& lhs, std::vector<U>& index) {
    if (index.size() != lhs.data.rows()) {
        std::cerr << "Number of index values does not match number of rows in data" << std::endl;
    }
    data_frame<U> df;
    df.set_data(lhs.get_data());
    df.set_column_names(lhs.get_column_names());
    df.assign_index(index);
    return df;
}
template <typename T>
void data_frame<T>::set_index(std::vector<T> index) {
if (index.size() != this->data.rows()) {
		std::cerr << "Number of index values does not match number of rows in data" << std::endl;
	}
	this->index = index;
}
//read fucntions must submit index column type
template <typename T>
data_frame<T> data_frame<T>::read_csv(std::string file_path, const int index_col) {
    data_frame tmp;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
    }
    // Count the number of rows and columns in the CSV file
    std::string line;
    int num_rows = 0;
    int num_cols = 0;
    std::vector<double> data_vec;
    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        int col = 0;
        while (std::getline(line_stream, line, ',')) {
            //add column names
            if (num_rows == 0) {
                if (col > index_col) {
                    tmp.column_names.push_back(line);
                    ++num_cols;
                }
                ++col;
                continue;
            }

            if (col == index_col) {
                tmp.index.push_back(line);
                ++col;
                continue;
            }

            if (col < index_col + 1) {
                ++col;
                continue;
            }

            data_vec.push_back(std::stod(line));
            ++col;
        }
        ++num_rows;

    }

    //remove column_names row
    num_rows -= 1;
    file.close();

    if (index_col == -1) {
        for (size_t i = 0; i < num_rows; ++i) {
            tmp.index.push_back(std::to_string(i));
        }
    }
    // Create an Eigen matrix to store the CSV data
    tmp.data = Eigen::Map<Eigen::MatrixXd>(data_vec.data(), num_cols, num_rows).transpose();
    return tmp;
}



//data view
template <typename T>
void data_frame<T>::head(const int n) const {
    int row_idx = 0;
    int col_idx = 0;
    //print column names
    int num_col = this->cols();
    int num_row = this->rows();
    std::cout << "Columns: ";
    while (col_idx < num_col) {
        std::cout << this->column_names[col_idx] << " ";
        ++col_idx;
    }
    std::cout << std::endl;
    while (row_idx < n and row_idx < num_row) {
        std::cout << this->index[row_idx] << " ";
        col_idx = 0;
        while (col_idx < num_col) {
            std::cout << this->data(row_idx, col_idx) << "  ";
            ++col_idx;
        }
        std::cout << std::endl;
        ++row_idx;
    }
}

template <typename T>
void data_frame<T>::tail(const int n) const {
    int row_idx = 1;
    int col_idx = 0;
    //print column names
    int num_col = data.cols();
    int num_row = data.rows();
    std::cout << "Columns: ";
    while (col_idx < num_col) {
        std::cout << this->column_names[col_idx] << " ";
        ++col_idx;
    }
    std::cout << std::endl;
    while (row_idx - 1 < n and row_idx - 1 > -1) {
        std::cout << this->index[num_row - row_idx] << " ";
        col_idx = 0;
        while (col_idx < num_col) {
            std::cout << this->data(num_row - row_idx, col_idx) << " ";
            ++col_idx;
        }
        std::cout << std::endl;
        ++row_idx;
    }
}

//data selection
//get columns basing on column names allowing non-exist columns which will be filled with nan
template <typename t>
data_frame<t> data_frame<t>::get_columns(const std::vector<std::string> _column_names) const {
    std::vector<int> column_indices;
    std::vector<std::string> col_names;
    std::vector<std::string> col_names_unfound;
    Eigen::MatrixXd data_copy = Eigen::MatrixXd::Constant(this->rows(), _column_names.size(), std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < _column_names.size(); ++i) {
        int col_idx = 0;
        while (col_idx < this->column_names.size()) {
            if (_column_names[i] == this->column_names[col_idx]) {
                data_copy.col(i) = this->data.col(col_idx);
                break;
            }
            ++col_idx;
        }
    }
    data_frame<t> data_frame_copy(data_copy, _column_names, this->index);
    return data_frame_copy;
}
template <typename t>
data_frame<t> data_frame<t>::get_columns(const std::vector<int> column_index) const {
    std::vector<std::string> column_names;
    for (size_t i = 0; i < column_index.size(); ++i) {
        column_names.push_back(this->column_names[column_index[i]]);
    }
    Eigen::MatrixXd data_copy = this->data(Eigen::all, column_index);
    data_frame data_frame_copy(data_copy, column_names, this->index);
    return data_frame_copy;
};

template <typename T>
data_frame<T> data_frame<T>::get_rows(const std::vector<T> sub_index) const {
    std::vector<int> index_indices;
    std::vector<T> valid_index_vec;
    for (size_t i = 0; i < sub_index.size(); i++) {
        auto ptr = std::find(this->index.begin(), this->index.end(), sub_index[i]);
        if (ptr != this->index.end()) {
            index_indices.push_back(ptr - this->index.begin());
            valid_index_vec.push_back(sub_index[i]);
        }
    }
    Eigen::MatrixXd data_copy = this->data(index_indices, Eigen::all);
    data_frame data_frame_copy(data_copy, this->column_names, valid_index_vec);
    return data_frame_copy;
}
template <typename T>
data_frame<T> data_frame<T>::get_rows(const std::vector<int> sub_index) const {
    Eigen::MatrixXd data_copy = this->data(sub_index, Eigen::all);
    std::vector<T> index_copy;
    for (size_t i = 0; i < sub_index.size(); ++i) {
        index_copy.push_back(this->index[sub_index[i]]);
    }
    data_frame data_frame_copy(data_copy, this->column_names, index_copy);
    return data_frame_copy;
}
template <typename T>
data_frame<T> data_frame<T>::get_rows(T index) const {
    size_t i;
    for (i = 0; i < this->index.size(); i++) {
        if (this->index[i] == index) {
            break;
        };
    };
    if (i == this->index.size()) {
        throw std::runtime_error("the target index is not in the data frame");
    }
    data_frame<T> data_frame_copy(this->data.row(i), this->column_names, { index });
    return data_frame_copy;
};

template <typename T>
data_frame<T> data_frame<T>::get_rows(int idx) const {
    if (idx > this->index.size()-1) {
        throw std::runtime_error("the target index is not in the data frame");
    }
    data_frame<T> data_frame_copy(this->data.row(idx), this->column_names, { this->index[idx]});
    return data_frame_copy;
};

template <typename T>
void data_frame<T>::insert_row(const int row_index, const T idx, const Eigen::MatrixXd& row) {
    //insert index
    this->index.insert(this->index.begin() + row_index, idx);
    //insert data
    this->data.conservativeResize(this->data.rows() + 1, this->data.cols());
    this->data.row(row_index) = row;
}

template <typename T>
data_frame<T> data_frame<T>::dropna(const int axis, const std::string how) {
    int num_row = this->data.rows();
    int num_col = this->data.cols();
    if (axis) {
        std::vector<int> valid_col_vec;
        int i;
        for (i = 0; i < num_col; ++i) {
            int nan_ct = 0;
            for (int j = 0; j < num_row; ++j) {
                if (std::isnan(this->data(j, i))) {
                    ++nan_ct;
                    if (how == "any") {
                        break;
                    }

                }
            }
            if ((how == "all" and nan_ct == num_row) or (nan_ct == 1 and how == "any")) {
                continue;
            }
            valid_col_vec.push_back(i);
        }
        data_frame<T> data_frame_copy = get_columns(valid_col_vec);
        return data_frame_copy;
    }
    else {

        std::vector<int> valid_row_vec;
        int i;
        for (i = 0; i < num_row; ++i) {
            int nan_ct = 0;
            for (int j = 0; j < num_col; j++) {
                if (std::isnan(this->data(i, j))) {
                    ++nan_ct;
                    if (how == "any") {
                        break;
                    }

                }
            }
            if ((how == "all" and nan_ct == num_col) or (nan_ct == 1 and how == "any")) {
                continue;
            }
            valid_row_vec.push_back(i);
        }
        data_frame<T> data_frame_copy = get_rows(valid_row_vec);
        return data_frame_copy;
    }
}

template <typename T>
double& data_frame<T>::operator()(const int row, const int col){
    return this->data(row, col);
};

template <typename T>
data_frame<T> data_frame<T>::left_join(data_frame<T>& df2) const {
    if (this->index.size() != df2.index.size()) {
        std::cerr << "Warning: row size not match, filled missing data with 0.0" << std::endl;
    }
    //left concat
    //employ hashset(unordered_set in CPP) to generate new data_frame

    std::vector<T> rhs_index = df2.get_index();
    std::unordered_set<T> index_set(rhs_index.begin(), rhs_index.end());
    //new data matrix
    Eigen::MatrixXd new_data(this->rows(), this->cols() + df2.cols());
    for (int i = 0; i < this->index.size(); i++) {
        if (index_set.count(this->index[i]) > 0) {
            new_data.row(i) << this->data.row(i), df2.get_rows(this->index[i]).get_data();
        }
        else {
            new_data.row(i) << this->data.row(i), Eigen::MatrixXd::Constant(1, df2.data.cols(), std::numeric_limits<double>::quiet_NaN());
        }
    }
  
    std::vector<std::string> column_names = this->column_names;
    column_names.insert(column_names.end(), df2.column_names.begin(), df2.column_names.end());
    data_frame data_frame_copy(new_data, column_names, this->index);
    return data_frame_copy;
}

template <typename T>
void data_frame<T>::row_concat(data_frame<T>& df2) {
    //empty df1
    if (this->data.size() == 0) {
        Eigen::MatrixXd tmp_data = df2.get_data();
        this->set_data(tmp_data);
        this->set_column_names(df2.get_column_names());
        this->set_index(df2.get_index());
    }
    //columns compare
    std::set<std::string> set1(this->column_names.begin(), this->column_names.end());
    std::vector<std::string> df2_col_names = df2.get_column_names();
    std::set<std::string> set2(df2_col_names.begin(), df2_col_names.end());
    std::vector<std::string> union_vec;
    std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(),union_vec.begin());
    //std::vector<std::string> union_vec(union_set.begin(), union_set.end());
    data_frame<T> new_col_df1 = this->get_columns(union_vec);
    data_frame<T> new_col_df2 = df2.get_columns(union_vec);

    Eigen::MatrixXd df1_mx = new_col_df1.get_data();
    Eigen::MatrixXd df2_mx = new_col_df2.get_data();
    //matrix row concat
    Eigen::MatrixXd new_data(df1_mx.cols(), df1_mx.rows() + df2_mx.rows());
    new_data << df1_mx.transpose(), df2_mx.transpose();
    new_data.transposeInPlace();
    this->index.insert(this->index.end(), new_col_df2.index.begin(), new_col_df2.index.end());
    this->data = new_data;
    this->column_names = union_vec;
}
//mathematical functions
template <typename T>
data_frame<T> data_frame<T>::log() {
    return data_frame<T>(this->data.array().log(), this->column_names, this->index);
};

template <typename T>
data_frame<T> data_frame<T>::exp() {
    return data_frame<T>(this->data.array().exp(), this->column_names, this->index);
};

template <typename T>
data_frame<T> data_frame<T>::row_diff() {
    Eigen::MatrixXd diffed_data(this->rows(), this->cols());
    for (int i = 0; i < this->data.rows() - 1; i++) {
        diffed_data.row(i + 1) = this->data.row(i + 1) - this->data.row(i);
    }
    diffed_data.row(0) = Eigen::MatrixXd::Constant(1, this->data.cols(), std::numeric_limits<double>::quiet_NaN());
    return data_frame<T>(diffed_data, this->column_names, this->index);
};