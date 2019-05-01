/*
 * Copyright (c) 2012-2013, 2015 ARM Limited
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
 *
 * Copyright (c) 2005 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Ron Dreslinski
 *          Steve Reinhardt
 */

/**
 * @file
 * This is a vanilla LSTM implementation.  It contains a forget, input, output, activation, and
 */

#include "mem/cache/prefetch/lstm.hh"
#include <ctime>
#include "base/random.hh"
#include "base/trace.hh"
#include "debug/HWPrefetch.hh"
#include <cstdlib>
#include "params/LSTMPrefetcher.hh"


LSTMPrefetcher::LSTMPrefetcher(const LSTMPrefetcherParams *p)
    : QueuedPrefetcher(p)
          {
    // Don't consult LSTM prefetcher on instruction accesses
    //from thesis paper
    learning_rate = .2;
    float_t init_weights_low = -.0001;
    float_t init_weights_hi = .0001;
    float_t hi = init_weights_low;
    float_t low =  init_weights_hi;
    srand (static_cast <unsigned> (time(0)));
    //initializing the LSTM parameters
    //initialze weights to random value between -10^-5 and 10^-5
    //from LSTM master thesis paper
    for(int i = 0; i< WEIGHT_SIZE_DIM; i++){
      for(int j =0; j < WEIGHT_SIZE_DIM; i++){


        weighti_i[i][j] = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi-low)));
        weighti_r[i][j] = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi-low)));
        weightf_i[i][j] = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi-low)));
        weightf_r[i][j] = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi-low)));
        weighto_i[i][j] = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi-low)));
        weighto_r[i][j] = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi-low)));
        weightz_i[i][j] = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi-low)));
        weightz_r[i][j] = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi-low)));
      }
    }
    //refer to An Empirical Exploration of Recurrent Network Architectures (Jozefowicz et al., 2015): for bias initialization

     biasi  = 1;
     biasf = 1;
     biaso = 1;
     biasz = 1;

    //initialze offset table states to 0
    for(int i = 0; i < OFFSET_TABLE_SIZE; i++){
      offset_table[i].state = 0;
    }
}

inline void
LSTMPrefetcher::backward_prop(Addr curr_pkt_addr){
  /*
  add 0 to dy for the 5th timestep since there are no future time steps
  */
//L2 LOSS FUNCTION -> pred - actual
int delta_t =

  return;


}



inline void
LSTMPrefetcher::matrix_product_forward_weights_int(double first[NUM_BLOCKS][INPUT_SIZE], int second[INPUT_SIZE], double result[INPUT_SIZE]){


    for(int i = 0; i < NUM_BLOCKS; i++){
            for(int k = 0; k < INPUT_SIZE; k++)
            {
                result[i] += first[i][k] * second[k];
            }
          }
    return;
}
inline void
LSTMPrefetcher::matrix_product_forward_weights_double(double first[NUM_BLOCKS][INPUT_SIZE], double second[INPUT_SIZE], double result[INPUT_SIZE]){


    for(int i = 0; i < NUM_BLOCKS; i++){
            for(int k = 0; k < INPUT_SIZE; k++)
            {
                result[i] += first[i][k] * second[k];
            }
          }
    return;
}


inline void
LSTMPrefetcher::outer_product_backprop(double first[INPUT_SIZE], int second[INPUT_SIZE], double result[INPUT_SIZE][INPUT_SIZE]){
  for(int i = 0; i < INPUT_SIZE; i++){
      for(int j = 0; j < INPUT_SIZE; j++){
          {
              result[i][j] = first[j] * second[i];
          }
        }
    }
        return;
}
//for both forward and backprop
inline void
LSTMPrefetcher::pointwise_product(double first[INPUT_SIZE], double second[INPUT_SIZE], double result[INPUT_SIZE]){

  for(int i = 0; i < INPUT_SIZE; ++i){

    result[i] = first[i] * second[i];


  }
        return;
}
inline void
LSTMPrefetcher::add_matrices(double first[INPUT_SIZE], int second[INPUT_SIZE], double result[INPUT_SIZE]){
  for(int i = 0; i < INPUT_SIZE; ++i){
    result[i] = first[i] + second[i];
  }
}
inline void
LSTMPrefetcher::add_matrices_double(double first[INPUT_SIZE], double second[INPUT_SIZE], double result[INPUT_SIZE]){
  for(int i = 0; i < INPUT_SIZE; ++i){
    result[i] = first[i] + second[i];
  }
}
inline void
LSTMPrefetcher::to_binary_array(int num, int array[INPUT_SIZE]){
  for(int i = 0; i < INPUT_SIZE; ++i){
    array[i] = (num >> i) & LSB_MASK;
  }
  return;
}




inline int
LSTMPrefetcher::from_array_to_num(int array[NUM_DELTA_BITS]){
  int result = 0;

  if(array[INPUT_SIZE-1]==0){
  for(int i = 0; i < INPUT_SIZE; ++i){
    if(i == 0 && array[i] == 1){
      result += 1;
    }
    else{
      if(array[i]==1){
    result += pow(2,i);
  }
  }

  }
}
  else{
    //first invert then add one
    for(int i = 0; i < INPUT_SIZE; ++i){
      if(array[i] == 1){
        array[i] =  0;
      }
      else{
        array[i] =  1;
      }
  }
  int i = 0;
   int carry=1;
   while(i<INPUT_SIZE&&carry==1)
   {
       if(array[i]==0)
       {
          array[i]=1;carry=0;
       }
       else
       {
           array[i]=0;carry=1;
       }
       i++;
   }
    for(int i = 0; i < INPUT_SIZE; ++i){
      if(i == 0 && array[i] == 1){
        result += 1;
      }
      else{
        if(array[i]==1){
          result += pow(2,i);
    }
    }
  }
  return -1 *result;
}
  return result;
}

inline void
LSTMPrefetcher::form_input(int page_num, int delta, int array[INPUT_SIZE]){
  for(int i = 0; i < INPUT_SIZE; ++i){
    if(i< INPUT_SIZE/2){
    array[i] = (delta >> i) & LSB_MASK;
      }
      else{
        array[i] = (page_num >> i) & LSB_MASK;
      }
  }
}

inline void
LSTMPrefetcher::stochastic_choice(int threshhold, double input[INPUT_SIZE], int result[INPUT_SIZE]){
  for(int i = 0; i < INPUT_SIZE; i++){
    if(input[i] > .5){
      result[i] = 1;
    }
    else{
      result[i] = 0;
    }
  }
}

inline void
LSTMPrefetcher::sigmoid(double first[INPUT_SIZE], double result[INPUT_SIZE]){

  for(int i = 0; i < INPUT_SIZE; ++i){
    result[i] = 1/(1+ exp(first[i]));
  }
}
inline void
LSTMPrefetcher::hyperbolicTanH(double first[INPUT_SIZE], double result[INPUT_SIZE]){
  for(int i = 0; i < INPUT_SIZE; ++i){
    result[i] = tanh(first[i]);
  }
}


inline void
LSTMPrefetcher::add_bias(int num, double first[INPUT_SIZE], double result[INPUT_SIZE]){
  for(int i = 0; i < INPUT_SIZE; ++i){
    result[i] = first[i] + num;
  }
}


inline void
LSTMPrefetcher::forward_prop(Addr pkt_addr){

//don't forget to change cache line size and use the makeline function? prob dont have to
  //use prestate for initial hidden state
  //index into the page table.  use the
  //form the input [x,y] where y is the hidden statefrom the previous timestep
  int x_t[INPUT_SIZE];
  double temp[INPUT_SIZE];
  double temp2[INPUT_SIZE];
  double temp_for_cell[INPUT_SIZE];
  int prediction[INPUT_SIZE];
  int processed_final_prediction[INPUT_SIZE];

  int curr_pgnum = (pkt_addr >> PAGE_NUMBER_SHIFT) & PAGE_NUMBER_MASK_FOR_PRED;
  int number_of_iterations = NUM_OF_DELTAS + 1;
  for(int i = 0; i < number_of_iterations; i++){
    if(i ==4){
      int input;
      input = (curr_pgnum <<NUM_PG_NUM_BITS) + curr_offset_delta;
      to_binary_array(input,x_t);

      //calculate using the previous state variables
      //block input calculation
      matrix_product_forward_weights_int(weightz_i, x_t,temp2);
      matrix_product_forward_weights_double(weightz_r, page_table[curr_page_table_idx].stateSpace.states[i-1].y,temp);
      add_matrices_double(temp2,temp,page_table[curr_page_table_idx].stateSpace.states[i].z_hat);
      add_bias(biasz,page_table[curr_page_table_idx].stateSpace.states[i].z_hat,page_table[curr_page_table_idx].stateSpace.states[i].z_hat);
      //add in stochastic choice
      hyperbolicTanH(page_table[curr_page_table_idx].stateSpace.states[i].z_hat, page_table[curr_page_table_idx].stateSpace.states[i].z);
      //add in gating

      //input gating
      matrix_product_forward_weights_int(weighti_i, x_t,temp2);
      matrix_product_forward_weights_double(weighti_r,page_table[curr_page_table_idx].stateSpace.states[i-1].y,temp);
      add_matrices_double(temp2,temp,page_table[curr_page_table_idx].stateSpace.states[i].i_hat);
      add_bias(biasi,page_table[curr_page_table_idx].stateSpace.states[i].i_hat,page_table[curr_page_table_idx].stateSpace.states[i].i_hat);
      //add in stochastic choice
      sigmoid(page_table[curr_page_table_idx].stateSpace.states[i].i_hat, page_table[curr_page_table_idx].stateSpace.states[i].i);

      //input gating
      matrix_product_forward_weights_int(weightf_i, x_t,temp2);
      matrix_product_forward_weights_double(weightf_r,page_table[curr_page_table_idx].stateSpace.states[i-1].y,temp);
      add_matrices_double(temp2,temp,page_table[curr_page_table_idx].stateSpace.states[i].f_hat);
      add_bias(biasf,page_table[curr_page_table_idx].stateSpace.states[i].f_hat,page_table[curr_page_table_idx].stateSpace.states[i].f_hat);
      //add in stochastic choice
      sigmoid(page_table[curr_page_table_idx].stateSpace.states[i].f_hat, page_table[curr_page_table_idx].stateSpace.states[i].f);

      //cell state calculations
      pointwise_product( page_table[curr_page_table_idx].stateSpace.states[i].z, page_table[curr_page_table_idx].stateSpace.states[i].i,temp);
      pointwise_product(page_table[curr_page_table_idx].stateSpace.states[i-1].c, page_table[curr_page_table_idx].stateSpace.states[i].f,temp_for_cell);
      add_matrices_double(temp, temp_for_cell,page_table[curr_page_table_idx].stateSpace.states[i].c);
      //add in stochastic choice

      //output gate
      matrix_product_forward_weights_int(weighto_i, x_t,temp2);
      matrix_product_forward_weights_double(weighto_r, page_table[curr_page_table_idx].stateSpace.states[i-1].y,temp);
      add_matrices_double(temp2,temp,page_table[curr_page_table_idx].stateSpace.states[i].o_hat);
      add_bias(biaso,page_table[curr_page_table_idx].stateSpace.states[i].o_hat,page_table[curr_page_table_idx].stateSpace.states[i].o_hat);
      //add in stochastic choice
      sigmoid(page_table[curr_page_table_idx].stateSpace.states[i].o_hat, page_table[curr_page_table_idx].stateSpace.states[i].o);


      //block output
      sigmoid(page_table[curr_page_table_idx].stateSpace.states[i].c, temp);
      pointwise_product(temp, page_table[curr_page_table_idx].stateSpace.states[i].o,page_table[curr_page_table_idx].stateSpace.states[i].y); //final prediction
      //stochastic choice
      stochastic_choice(THRESHHOLD, page_table[curr_page_table_idx].stateSpace.states[i].y,prediction);
      //serpeate the delta from the page number
      for(int i = 0; i < NUM_DELTA_BITS; i++){
        processed_final_prediction[i] = prediction[i];
      }

      //update the offset table
      final_prediction_delta = from_array_to_num(processed_final_prediction);
      //still have to add in the rest of the bits
    }
    //calculate using the delta stored in the table associated with the ith timestep where i is also the index
    else{

      //calculate using preState
      if(i == 0){
        //calculate delta from table
        int input;
        input = (curr_pgnum <<NUM_PG_NUM_BITS) + page_table[curr_pgnum].prevDeltas[i];
        to_binary_array(input,x_t);

      //block input calculation
      matrix_product_forward_weights_int(weightz_i, x_t,temp2);
      matrix_product_forward_weights_int(weightz_r, page_table[curr_page_table_idx].preState.hidden_state,temp);
      add_matrices_double(temp2,temp,page_table[curr_page_table_idx].stateSpace.states[i].z_hat);
      add_bias(biasz,page_table[curr_page_table_idx].stateSpace.states[i].z_hat,page_table[curr_page_table_idx].stateSpace.states[i].z_hat);
      //add in stochastic choice
      hyperbolicTanH(page_table[curr_page_table_idx].stateSpace.states[i].z_hat, page_table[curr_page_table_idx].stateSpace.states[i].z);
      //add in gating

      //input gating
      matrix_product_forward_weights_int(weighti_i, x_t,temp2);
      matrix_product_forward_weights_int(weighti_r, page_table[curr_page_table_idx].preState.hidden_state,temp);
      add_matrices_double(temp2,temp,page_table[curr_page_table_idx].stateSpace.states[i].i_hat);
      add_bias(biasi,page_table[curr_page_table_idx].stateSpace.states[i].i_hat,page_table[curr_page_table_idx].stateSpace.states[i].i_hat);
      //add in stochastic choice
      sigmoid(page_table[curr_page_table_idx].stateSpace.states[i].i_hat, page_table[curr_page_table_idx].stateSpace.states[i].i);

      //input gating
      matrix_product_forward_weights_int(weightf_i, x_t,temp2);
      matrix_product_forward_weights_int(weightf_r, page_table[curr_page_table_idx].preState.hidden_state,temp);
      add_matrices_double(temp2,temp,page_table[curr_page_table_idx].stateSpace.states[i].f_hat);
      add_bias(biasf,page_table[curr_page_table_idx].stateSpace.states[i].f_hat,page_table[curr_page_table_idx].stateSpace.states[i].f_hat);
      //add in stochastic choice
      sigmoid(page_table[curr_page_table_idx].stateSpace.states[i].f_hat, page_table[curr_page_table_idx].stateSpace.states[i].f);

      //cell state calculations
      pointwise_product( page_table[curr_page_table_idx].stateSpace.states[i].z, page_table[curr_page_table_idx].stateSpace.states[i].i,temp);
      pointwise_product(page_table[curr_page_table_idx].preState.cell_state,page_table[curr_page_table_idx].stateSpace.states[i].f,temp_for_cell);
      add_matrices_double(temp, temp_for_cell,page_table[curr_page_table_idx].stateSpace.states[i].c);
      //add in stochastic choice

      //output gate
      matrix_product_forward_weights_int(weighto_i, x_t,temp2);
      matrix_product_forward_weights_int(weighto_r, page_table[curr_page_table_idx].preState.hidden_state,temp);
      add_matrices_double(temp2,temp,page_table[curr_page_table_idx].stateSpace.states[i].o_hat);
      add_bias(biaso,page_table[curr_page_table_idx].stateSpace.states[i].o_hat,page_table[curr_page_table_idx].stateSpace.states[i].o_hat);
      //add in stochastic choice
      sigmoid(page_table[curr_page_table_idx].stateSpace.states[i].o_hat, page_table[curr_page_table_idx].stateSpace.states[i].o);


      //block output
      sigmoid(page_table[curr_page_table_idx].stateSpace.states[i].c, temp);
      pointwise_product(temp, page_table[curr_page_table_idx].stateSpace.states[i].o,page_table[curr_page_table_idx].stateSpace.states[i].y);
      //stochastic choice

    }
    else{

      //calculate using the previous state variables
      int input;
      input = (curr_pgnum <<NUM_PG_NUM_BITS) + page_table[curr_pgnum].prevDeltas[i];
      to_binary_array(input,x_t);

      //block input calculation
      matrix_product_forward_weights_int(weightz_i, x_t,temp2);
      matrix_product_forward_weights_double(weightz_r, page_table[curr_page_table_idx].stateSpace.states[i-1].y,temp);
      add_matrices_double(temp2,temp,page_table[curr_page_table_idx].stateSpace.states[i].z_hat);
      add_bias(biasz,page_table[curr_page_table_idx].stateSpace.states[i].z_hat,page_table[curr_page_table_idx].stateSpace.states[i].z_hat);
      //add in stochastic choice
      hyperbolicTanH(page_table[curr_page_table_idx].stateSpace.states[i].z_hat, page_table[curr_page_table_idx].stateSpace.states[i].z);
      //add in gating

      //input gating
      matrix_product_forward_weights_int(weighti_i, x_t,temp2);
      matrix_product_forward_weights_double(weighti_r,page_table[curr_page_table_idx].stateSpace.states[i-1].y,temp);
      add_matrices_double(temp2,temp,page_table[curr_page_table_idx].stateSpace.states[i].i_hat);
      add_bias(biasi,page_table[curr_page_table_idx].stateSpace.states[i].i_hat,page_table[curr_page_table_idx].stateSpace.states[i].i_hat);
      //add in stochastic choice
      sigmoid(page_table[curr_page_table_idx].stateSpace.states[i].i_hat, page_table[curr_page_table_idx].stateSpace.states[i].i);

      //input gating
      matrix_product_forward_weights_int(weightf_i, x_t,temp2);
      matrix_product_forward_weights_double(weightf_r,page_table[curr_page_table_idx].stateSpace.states[i-1].y,temp);
      add_matrices_double(temp2,temp,page_table[curr_page_table_idx].stateSpace.states[i].f_hat);
      add_bias(biasf,page_table[curr_page_table_idx].stateSpace.states[i].f_hat,page_table[curr_page_table_idx].stateSpace.states[i].f_hat);
      //add in stochastic choice
      sigmoid(page_table[curr_page_table_idx].stateSpace.states[i].f_hat, page_table[curr_page_table_idx].stateSpace.states[i].f);

      //cell state calculations
      pointwise_product( page_table[curr_page_table_idx].stateSpace.states[i].z, page_table[curr_page_table_idx].stateSpace.states[i].i,temp);
      pointwise_product(page_table[curr_page_table_idx].stateSpace.states[i-1].c, page_table[curr_page_table_idx].stateSpace.states[i].f,temp_for_cell);
      add_matrices_double(temp, temp_for_cell,page_table[curr_page_table_idx].stateSpace.states[i].c);
      //add in stochastic choice

      //output gate
      matrix_product_forward_weights_int(weighto_i, x_t,temp2);
      matrix_product_forward_weights_double(weighto_r, page_table[curr_page_table_idx].stateSpace.states[i-1].y,temp);
      add_matrices_double(temp2,temp,page_table[curr_page_table_idx].stateSpace.states[i].o_hat);
      add_bias(biaso,page_table[curr_page_table_idx].stateSpace.states[i].o_hat,page_table[curr_page_table_idx].stateSpace.states[i].o_hat);
      //add in stochastic choice
      sigmoid(page_table[curr_page_table_idx].stateSpace.states[i].o_hat, page_table[curr_page_table_idx].stateSpace.states[i].o);


      //block output
      sigmoid(page_table[curr_page_table_idx].stateSpace.states[i].c, temp);
      pointwise_product(temp, page_table[curr_page_table_idx].stateSpace.states[i].o,page_table[curr_page_table_idx].stateSpace.states[i].y);
      //stochastic choice


    }
    }
  }


  return;
}
inline void
LSTMPrefetcher::new_page_entry(){

  page_table_entry new_entry = {
    curr_page_num,
    curr_prevAddr,
    0,
    {DEFAULT_DELTA, DEFAULT_DELTA, DEFAULT_DELTA, DEFAULT_DELTA}, //
    {DEFAULT_PREVADDR, DEFAULT_PREVADDR, DEFAULT_PREVADDR, DEFAULT_PREVADDR}, //so addresses don't match for prediction
    0,
    {
      //state space initialization

      { {
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0}
    }
    ,
       {
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0}
    }
    ,
       {
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0}
    }
    ,
       {
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0}

    }
  }
  },

    {
      {0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0}
    }


  };


  page_table.insert(page_table.begin(), new_entry);

  return;
}
//if the offset matches an offset in pre4fetchaddrs and there are enough deltas
//then predict
//otherwise evict or update an entry
inline bool
LSTMPrefetcher::check_page_table(){
  curr_page_table_idx = -1;
  for(int i = 0; i < PAGE_TABLE_SIZE; i++){
    if(page_table[i].pageNumber == curr_page_num){
      curr_page_table_idx = i;
      break;
    }
  }
  if(curr_page_table_idx == -1){ //not in the page table
    if(page_table.size() < 64){
    //allocate new entry
    new_page_entry();
    return 0;
    }
    else{
        //pop back the LRU and insert new node
          page_table.pop_back();
          new_page_entry();
          return 0;
    }
  }
  else{ //we found an entry that matches
    //move to MRU position or front

    page_table.insert(page_table.begin(), page_table[curr_page_table_idx]);
    page_table.erase(page_table.begin() + curr_page_table_idx + 1);
    int delta_count = 0;
    int offset_match = 0;
    for(int i = 0; i < NUM_OF_DELTAS; i++){
      // check if there are enough deltas to make a prediction and if the offset exists
      if(page_table[curr_page_table_idx].prevDeltas[i] == DEFAULT_DELTA){
        delta_count++;
      }
      if(page_table[curr_page_table_idx].prev4Addrs[i] == curr_offset){
        offset_match = 1;
      }
    }
    if(delta_count < 3){
      return 0;
    }
    else if(offset_match){
      return 1;
    }
      else{
        return 0;
      }
    }
}



inline bool //checks offset table and page table
LSTMPrefetcher::checkoffset(Addr pkt_addr){
  //first find the page offset entry

  if(offset_table[curr_offset].state == 0){
    offset_table[curr_offset].state = 1;
    offset_table[curr_offset].first_address = pkt_addr;
    return 0; //no prediction
  }
  else if(offset_table[curr_offset].state == 1){
    offset_table[curr_offset].state = 2;
    offset_table[curr_offset].delta = ((pkt_addr >> DELTA_SHIFT) & DELTA_MASK) - ((offset_table[curr_offset].first_address >> DELTA_SHIFT) & DELTA_MASK);
    curr_offset_delta = offset_table[curr_offset].delta;
    return 0; //no prediction
  }
  else{
    curr_offset_delta = offset_table[curr_offset].delta;
    return 1; // predict using delta if first access to page
  }
}

void
LSTMPrefetcher::calculatePrefetch(const PrefetchInfo &pfi,
                       std::vector<AddrPriority> &addresses){
     if (!pfi.hasPC()) {
         DPRINTF(HWPrefetch, "Ignoring request with no PC.\n");
         return;
     }


    // Get required packet info
    Addr pkt_addr = pfi.getAddr();

    //MasterID master_id = useMasterId ? pkt->req->masterId() : 0;

    //populate current variables used in prediction

    curr_offset = (pkt_addr & PAGE_OFFSET_MASK);
    curr_page_num = (pkt_addr >> PAGE_NUMBER_SHIFT) & PAGE_NUMBER_MASK_FOR_PRED;
    curr_prevAddr = (pkt_addr >> PREV_ADDR_SHIFT) & PREV_ADDR_MASK;

    if (checkoffset(pkt_addr)) {
        //evict or update page table entry
        if(check_page_table()){
          //found a offset hit in prev4fetchAddrs and do a prediction with the four deltas
          //update weights before prediction
            backward_prop(pkt_addr);
            forward_prop(pkt_addr);

            Addr new_addr = (curr_prevAddr << PREV_ADDR_SHIFT) + (final_prediction_delta<< DELTA_SHIFT);


            //update page_table
            int idx;
            for(int i = 0; i < NUM_OF_DELTAS-1; i++){
            idx = 3-i;
            page_table[curr_page_table_idx].prevDeltas[idx] = page_table[curr_page_table_idx].prevDeltas[idx-1];
            page_table[curr_page_table_idx].prev4Addrs[idx] = page_table[curr_page_table_idx].prevDeltas[idx-1];

          }
          //update the first entry
          page_table[curr_page_table_idx].prevDeltas[0] = curr_offset_delta;
          page_table[curr_page_table_idx].prev4Addrs[0] = final_prediction_delta;

            if (samePage(pkt_addr, new_addr)) {
                DPRINTF(HWPrefetch, "Queuing prefetch to %#x.\n", new_addr);
                addresses.push_back(AddrPriority(new_addr, 0));
            } else {
                // Record the number of page crossing prefetches generated
                DPRINTF(HWPrefetch, "Ignoring page crossing prefetch.\n");
                return;
            }

          }
          else{  //first access to a page so use the delta in the offset table
            // Miss in page table
            //push on addr using delta from offset table
            Addr new_addr = (curr_prevAddr << PREV_ADDR_SHIFT) + (curr_offset_delta << DELTA_SHIFT);
            //update page_table
            page_table[curr_page_table_idx].prevDeltas[0] = curr_offset_delta;
            page_table[curr_page_table_idx].prev4Addrs[0] = curr_offset_delta; // curr_offset;
            if (samePage(pkt_addr, new_addr)) {
                DPRINTF(HWPrefetch, "Queuing prefetch to %#x.\n", new_addr);
                addresses.push_back(AddrPriority(new_addr, 0));
            } else {
                // Record the number of page crossing prefetches generated
                DPRINTF(HWPrefetch, "Ignoring page crossing prefetch.\n");
                return;
            }


          }

    } else {

      //miss in offset table
      //updated offest table in previous call
      //update page_table
      bool ignore = check_page_table(); //this will insert new entry if necessary
      if(ignore){
        return;
      }
      return;
    }
}



LSTMPrefetcher*
LSTMPrefetcherParams::create()
{
    return new LSTMPrefetcher(this);
}
