#include <iostream>
#include <string>
#include <vector>

using namespace std;

enum class TensorOp {
    Scalar,
    Var,
    Add,
    Mult,
    Transpose,
    Dot,
    Inverse
};

template <typename A, typename B>
struct Tensor {
    TensorOp op;
    A arg1;
    B arg2;

    Tensor(TensorOp op, A arg1, B arg2) : op(op), arg1(arg1), arg2(arg2) {}
};

template <typename A>
struct Tensor<A, void> {
    TensorOp op;
    A arg1;

    Tensor(TensorOp op, A arg1) : op(op), arg1(arg1) {}
};

template <typename T>
struct Scalar {
    T value;

    Scalar(T value) : value(value) {}
};

template <typename T>
struct Var {
    string name;

    Var(string name) : name(name) {}
};

template <typename A, typename B>
Tensor<A, B> make_tensor(TensorOp op, A arg1, B arg2) {
    return Tensor<A, B>(op, arg1, arg2);
}

template <typename A>
Tensor<A, void> make_tensor(TensorOp op, A arg1) {
    return Tensor<A, void>(op, arg1);
}

template <typename T>
Scalar<T> make_scalar(T value) {
    return Scalar<T>(value);
}

Var<double> make_var(string name) {
    return Var<double>(name);
}

template <typename T>
ostream& operator<<(ostream& os, const Scalar<T>& scalar) {
    os << scalar.value;
    return os;
}

ostream& operator<<(ostream& os, const Var<double>& var) {
    os << var.name;
    return os;
}

template <typename A, typename B>
ostream& operator<<(ostream& os, const Tensor<A, B>& tensor) {
    switch (tensor.op) {
        case TensorOp::Scalar:
            os << make_scalar(tensor.arg1);
            break;
        case TensorOp::Var:
            os << make_var(tensor.arg1);
            break;
        case TensorOp::Add:
            os << "(" << tensor.arg1 << " + " << tensor.arg2 << ")";
            break;
        case TensorOp::Mult:
            os << "(" << tensor.arg1 << " * " << tensor.arg2 << ")";
            break;
        case TensorOp::Transpose:
            os << "(" << tensor.arg1 << ").transpose()";
            break;
        case TensorOp::Dot:
            os << "(" << tensor.arg1 << ").dot(" << tensor.arg2 << ")";
            break;
        case TensorOp::Inverse:
            os << "(" << tensor.arg1 << ").inverse()";
            break;
    }
    return os;
}

class TensorParser {
public:
    Tensor<double, void> parse(string expression) {
        tokens.clear();
        tokenize(expression);

        idx = 0;
        auto tensor = parse_tensor();

        return tensor;
    }

private:
    void tokenize(string expression) {
        size_t
