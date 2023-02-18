### Introduction


Specification in tensors 
<!-- TOC -->
    * [Introduction](#introduction)
    * [References:](#references-)
<!-- TOC -->

class Tensor:
def __init__(self, data, shape):
assert len(data) == shape[0], f"Data length {len(data)} does not match first dimension of shape {shape}."
self.data = data
self.shape = shape

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[index]
        elif isinstance(index, tuple):
            result = self.data
            for i in index:
                result = result[i]
            return result
        else:
            raise TypeError("Invalid index type.")

    def __setitem__(self, index, value):
        if isinstance(index, int):
            self.data[index] = value
        elif isinstance(index, tuple):
            result = self.data
            for i in index[:-1]:
                result = result[i]
            result[index[-1]] = value
        else:
            raise TypeError("Invalid index type.")


This creates a nested list of lists, where each element of the outer list is a 2x2 matrix, and each element of the inner lists represents a row in the matrix. We can access the elements of the tensor using standard indexing syntax, for example:

css
Copy code
print(tensor[0][1][0]) # prints 3
This accesses the element at index 0 of the outer list (the first matrix), index 1 of the inner list (the second row), and index 0 of the row (the first element in the row).





Constants

|     |     |     |     |
|-----|-----|-----|-----|
|     |     |     |     |

Tensor


struct TF_Tensor {
TF_DataType dtype;
TensorShape shape;
TensorBuffer* buffer;
};

Buffer 



Shape 


Work flow

Tensor -> Deconde - endode strings ->  ApiTensor 

## Models in research:
 

## Tensorflow
https://github.com/openai/CLIP-featurevis/blob/master/tokenizer.py



Types


DoubleTensor

FloatTensor

IntTensor

ShortTensor


CharTensor

ByteTensor

### References:

https://github.com/tensorflow/tensorflow/commit/f41959ccb2d9d4c722fe8fc3351401d53bcf4900
