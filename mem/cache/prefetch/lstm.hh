
/**
 * @file
 * LSTM Prefetcher
 Based off of the work from Lehigh University by Yuan
 */

#ifndef __MEM_CACHE_PREFETCH_LSTM_HH__
#define __MEM_CACHE_PREFETCH_LSTM_HH__

#include <list>
#include <cmath>
#include <unordered_map>

#include "mem/cache/prefetch/queued.hh"
#include <vector>
#include <algorithm>
#define OFFSET_TABLE_SIZE 64
#define PAGE_TABLE_SIZE 64
#define INPUT_SIZE 13
#define NUM_BLOCKS 13
#define PAGE_OFFSET_MASK 63
#define DELTA_SHIFT 6
#define DELTA_MASK  127
#define PAGE_NUMBER_SHIFT 12
#define PAGE_NUMBER_MASK_FOR_PRED 255
#define PAGE_NUMBER_MASK_FOR_FIN_PRED 4095
#define PREV_ADDR_SHIFT 12
#define PREV_ADDR_MASK 4095
#define DEFAULT_DELTA 65
#define DEFAULT_PREVADDR -1
#define NUM_OF_DELTAS 4
#define WEIGHT_SIZE_DIM 13
#define NUM_BLOCKS 13
#define INPUT_SIZE 13
#define LSB_MASK 1 //history length
#define THRESHHOLD .5 //for stochastic choice
#define NUM_DELTA_BITS 7 //number of bits used for delta
#define NUM_PG_NUM_BITS 6 //number of bits used for page number
#define BACKPROP_INPUT_MASK 8191 //number of bits used for page number


struct LSTMPrefetcherParams;
class LSTMPrefetcher : public QueuedPrefetcher
{
  protected:
//assumptions
//try various page number configurations 8/7/6 bits
//also don't forget to check the is_secure and masterid
//change blck size to 64 bytes
//does the page number set up seem correct?  How do they calculate the next address?  What is the delta the page offset?
//change the memory block size size
//offset is the first 6 bits
//don't know if page table updates using the vector functinos are correct
//my assumption is that the delta is the page offset that she refers to and you use the prevAddr as for the next addr calculation
//default cache line size is 64 kb
//assume the same page number as origicanl address
//should I do a hashing for the page numbers?  from 8 to
//CHECK all syntax even little things like ++
//vars for the final address calculation
//make sure array access in back and forward are correct
//check transpose


    int curr_offset_delta;
    int curr_delta;
    int curr_page_num;
    int curr_offset;
    int curr_prevAddr;
    int prev_prevAddr;
    int final_prediction_delta;
    int curr_page_table_idx;


    double learning_rate;
    int prediction;
    double cell_state;
    double hidden_state;
    int block_input[INPUT_SIZE];

//backprop variables

    int first_delta_y[INPUT_SIZE];
    double delta_y[INPUT_SIZE];
    double delta_o_hat[INPUT_SIZE];
    double delta_c[INPUT_SIZE];
    double delta_f_hat[INPUT_SIZE]; //change these later
    double delta_i_hat[INPUT_SIZE];
    double delta_z_hat[INPUT_SIZE];


    struct offset_table_entry{
      int delta;
      int first_address;  //can also be obtained from the page table
      //state 0 means there is no match in the offset table (i.e. first access)
      //state 2 means that it hsa been accessed twice and holds a delta
      int state;
    };



    struct state{
      double cell_state [INPUT_SIZE];
      int hidden_state [INPUT_SIZE];
    };
    struct forward_pass_args{
      int x_t[INPUT_SIZE];
      double z_hat [INPUT_SIZE]; //block input
      double z [INPUT_SIZE];
      double i_hat [INPUT_SIZE]; //input gate
      double i [INPUT_SIZE];
      double f_hat  [INPUT_SIZE]; //forget
      double f [INPUT_SIZE];
      double c [INPUT_SIZE]; //cell state
      double o_hat [INPUT_SIZE]; //oyutput gate
      double o [INPUT_SIZE];
      double y [INPUT_SIZE]; //block output or prediction

    };
    struct state_space {
      forward_pass_args states[NUM_OF_DELTAS+1]; //for the final calculation since we use a history of 4
    };

    struct page_table_entry{

      int pageNumber; //7 bits
      int preAddr; //20 bits
      int pageSign; //6 bits
      int prevDeltas[NUM_OF_DELTAS]; //7 bit twos complement
      int prev4Addrs[NUM_OF_DELTAS]; // 4 * 6 bits
      int accuracy; // 1 bit
      state_space stateSpace; // hidden and cell states
      state preState; // ^
    };

    //pageoffset table
    //indexed by the page offset or bits [11:5]
    //A shared global array for first predictions, it holds the
    //first page offset and delta.

    offset_table_entry offset_table[OFFSET_TABLE_SIZE];

    //local page table for delta history for predictions.
    //also holds information for updating weights
    std::vector<page_table_entry> page_table;

private:
      //i is the input weights and r is the recurrent weights
      double weighti_i[NUM_BLOCKS][INPUT_SIZE];
      double weighti_r[NUM_BLOCKS][INPUT_SIZE];
      double weightf_i[NUM_BLOCKS][INPUT_SIZE];
      double weightf_r[NUM_BLOCKS][INPUT_SIZE];
      double weighto_i[NUM_BLOCKS][INPUT_SIZE];
      double weighto_r[NUM_BLOCKS][INPUT_SIZE];
      double weightz_i[NUM_BLOCKS][INPUT_SIZE];
      double weightz_r[NUM_BLOCKS][INPUT_SIZE];
      double biasi[NUM_BLOCKS];
      double biasf[NUM_BLOCKS];
      double biaso[NUM_BLOCKS];
      double biasz[NUM_BLOCKS];


    //check "A Space Odyssey" for LSTM formulas

    //offset table and page table lookup on Prefetch activation event
    bool checkoffset(Addr pkt_addr);
    bool check_page_table();

    //equations and variable names taken from "LSTM: Aint Search Space Odyssey"
    void forward_prop(Addr pkt_addr);
    void backward_prop(Addr pkt_addr);
    void evict_LRU(Addr pkt_addr);
    void update_offset_table(int offset, Addr pkt_addr);
    void new_page_entry();

    //operations


    double fRand(double fMin, double fMax);
      //functions for 2d arrays
      // R1 and C1 is the first operand for all functions
    //weights are N by M where N is the number of blocks and M is the input size

    void matrix_product_forward_weights_int(double first[NUM_BLOCKS][INPUT_SIZE], int second[NUM_BLOCKS], double result[INPUT_SIZE]);
    void int_to_double_array(int first[NUM_BLOCKS], double result[INPUT_SIZE]);
    void matrix_product_forward_weights_double(double first[NUM_BLOCKS][INPUT_SIZE], double second[NUM_BLOCKS], double result[INPUT_SIZE]);
    void outer_product_backprop(double first[INPUT_SIZE], int second[INPUT_SIZE], double result[INPUT_SIZE][INPUT_SIZE]);
    void outer_product_double(double first[INPUT_SIZE], double second[INPUT_SIZE], double result[INPUT_SIZE][INPUT_SIZE]);
    //for both forward and backprop
    void pointwise_product(double first[INPUT_SIZE], double second[INPUT_SIZE], double result[INPUT_SIZE]);
    void pointwise_product_int(double first[INPUT_SIZE], int second[INPUT_SIZE], double result[INPUT_SIZE]);
    void add_matrices(double first[INPUT_SIZE], int second[INPUT_SIZE], double result[INPUT_SIZE]);
    void add_matrices_double(double first[INPUT_SIZE], double second[INPUT_SIZE], double result[INPUT_SIZE]);
    void add_two_d_matrices_double(double first[INPUT_SIZE][INPUT_SIZE], double second[INPUT_SIZE][INPUT_SIZE], double result[INPUT_SIZE][INPUT_SIZE]);
    void scalar_multiply_two_by_two_matrix(double scalar, double first[INPUT_SIZE][INPUT_SIZE],  double result[INPUT_SIZE][INPUT_SIZE]);
    void transpose_back_prop(double first[INPUT_SIZE][INPUT_SIZE],  double result[INPUT_SIZE][INPUT_SIZE]);
    void to_binary_array(int num, int array[INPUT_SIZE]); //works
    int from_array_to_num(int array[INPUT_SIZE]);
    void form_input(int page_num, int delta, int arrary[INPUT_SIZE]);
    void stochastic_choice(int threshhold, double input[INPUT_SIZE],int result[INPUT_SIZE]);
    void sigmoid(double first[INPUT_SIZE], double result[INPUT_SIZE]);
    void dydx_sigmoid(double first[INPUT_SIZE], double result[INPUT_SIZE]);
    void dydx_hyperbolicTanH(double first[INPUT_SIZE], double result[INPUT_SIZE]);
    void hyperbolicTanH(double first[INPUT_SIZE], double result[INPUT_SIZE]);
    void add_bias(double first[INPUT_SIZE], double second[INPUT_SIZE], double result[INPUT_SIZE]);


  public:

    LSTMPrefetcher(const LSTMPrefetcherParams *p);

    void calculatePrefetch(const PrefetchInfo &pfi,
                           std::vector<AddrPriority> &addresses) override;
};

#endif // __MEM_CACHE_PREFETCH_LSTM_HH__
