




#include "LSTM_ops.hh"
void matrix_product_forward_weights(int first[NUM_BLOCKS][INPUT_SIZE], int second[INPUT_SIZE], int result[INPUT_SIZE]){


    for(int i = 0; i < NUM_BLOCKS; ++i){
        for(int j = 0; j < NUM_BLOCKS; ++j){
            for(int k = 0; k < INPUT_SIZE; ++k)
            {
                result[i] += first[i][k] * second[k];
            }
          }
        }

    return;
}


void outer_product_backprop(int first[INPUT_SIZE], int second[INPUT_SIZE], int result[INPUT_SIZE][INPUT_SIZE]){



  for(int i = 0; i < INPUT_SIZE; ++i){
      for(int j = 0; j < INPUT_SIZE; ++j){
          {
              result[i][j] = first[j] * second[i];
          }
        }
    }
        return;
}
//for both forward and backprop
void pointwise_product(int first[INPUT_SIZE], int second[INPUT_SIZE], int result[INPUT_SIZE]){

  for(int i = 0; i < INPUT_SIZE; ++i){

    result[i] = first[i] * second[i];


  }
        return;
}
void add_matrices(int first[INPUT_SIZE], int second[INPUT_SIZE], int result[INPUT_SIZE]){
  for(int i = 0; i < INPUT_SIZE; ++i){

    result[i] = first[i] + second[i];


  }

}
void to_binary_array(int num, int array[INPUT_SIZE]){

  for(int i = 0; i < INPUT_SIZE; ++i){

    array[i] = (num >> i) & LSB_MASK;


  }

}
int from_array_to_num(int array[INPUT_SIZE]){
  int result = 0;
  for(int i = 0; i < INPUT_SIZE; ++i){
    if(i == 0 && array[i] == 1){
      result += 1;
    }
    else{
      if(array[i]==1){
    result += 2*i;
  }
  }

  }
  return result;
}

void form_input(int page_num, int delta, int arrary[INPUT_SIZE]){
  for(int i = 0; i < INPUT_SIZE; ++i){
    if(i<)
    array[i] = (num >> i) & LSB_MASK;


  }

}

void stochastic_choice(int threshhold, double input[INPUT_SIZE]){

  return;
}
