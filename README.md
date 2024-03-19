# COGNTIV Code Assigment



## Installation

install python 3.9.5

install requirements specified in requirements.txt
```bash
pip install -r requirements.txt
```

## Usage

Create an example Conv-block
```bash
python robust_conv_block/scripts/conv_block_example.py
```


## Summary

I have incorporated the following modifications organized under the following categories:

### Parameters and Arguments:
   
1. Type hinting all functions and classes
2. Use Enums to declare the legal values of each operation / activation weight initialization
3. add assertions and checks to ensure that the input parameters are valid (e.g. that the tensor dimension is either 1 or 2)
4. Convert the type of the conv_block input to be a sequence of Operations, instead of a string, which removes the need to parse the string to the necessary operations, allows type hinting in the code, and makes for a better and safer practice overall. 

### Code Sturcture and Readability:
1. Use of docstrings to describe the purpose of the main functions and classes
2. Split the construction of the conv block into smaller functions to improve readability
3. Use different classes for the 1D and 2D convolutions, while using shared methods for similar logic (as in the separable convolution). This removes the need to keep track on whether the parameters are valid for a 2D conv or a 1D conv in the same class.
4. remove redundant code (as in the case of the unnecessary padding logic when causal == False)
5. create a conv_layer factory to create the correct conv layer based on the given parameters. This makes it easier to handle invalid arguments outside of the ConvLayer class, and enables more flexibility for constructing various versions of the ConvLayer. 


### Bug Fixes and Functionality:
1. Modified the code such that the current in_channels is maintained to accommodate any following operation
2. Implementd a 1D and a 2D version of Causal Convolution. The 2D version is based on a Masked Convolution, which is a better approach than using padding and cropping of the output
3. Implemented a Seperable Convolution, which can also be Causal. 


### Tests:
1. Implemented a few elementary tests to test the functionality of the conv block.  
2. All tests are parameterized using fixtures, which makes it easier to add more tests in the future.


### Backlog:
1. Implement more tests to cover the different functionalities of the conv block
2. allow for multiple types of acitvations and weight initializations in the same block
3. allow for different parameters for the convolutional layers in the same block
4. Support 3D tensors



