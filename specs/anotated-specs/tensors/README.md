### Introduction


Specification in tensors 
<!-- TOC -->
    * [Introduction](#introduction)
    * [References:](#references-)
<!-- TOC -->


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
