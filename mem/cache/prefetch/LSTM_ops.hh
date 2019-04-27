//matrix operations

//inner product
#ifndef __LSTM_OPS_HH__
#define __LSTM_OPS_HH__

#include <list>

#define NUM_BLOCKS 13
#define INPUT_SIZE 13
#define LSB_MASK 1


  //functions for 2d arrays
  // R1 and C1 is the first operand for all functions
//weights are N by M where N is the number of blocks and M is the input size
void matrix_product_forward_weights(double first[NUM_BLOCKS][INPUT_SIZE], int second[NUM_BLOCKS], double result[INPUT_SIZE]);
void outer_product_backprop(double first[INPUT_SIZE], double second[INPUT_SIZE], double result[INPUT_SIZE][INPUT_SIZE]);
//for both forward and backprop
void pointwise_product(double first[INPUT_SIZE], double second[INPUT_SIZE], int result[INPUT_SIZE]);
void add_matrices(double first[INPUT_SIZE], double second[INPUT_SIZE], double result[INPUT_SIZE]);
void to_binary_array(int num, int array[INPUT_SIZE]);
int from_array_to_num(int array[INPUT_SIZE]);
void form_input(int page_num, int delta, int arrary[INPUT_SIZE]);
void stochastic_choice(int threshhold, double input[INPUT_SIZE]);
void sigmoid(double first[INPUT_SIZE], double result[INPUT_SIZE]);
void tanh(double first[INPUT_SIZE], double result[INPUT_SIZE]);
void add_bias(int num, double first[INPUT_SIZE], double result[INPUT_SIZE]);

#endif
