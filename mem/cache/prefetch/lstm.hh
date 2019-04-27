
/**
 * @file
 * LSTM Prefetcher
 Based off of the work from Lehigh University by Yuan
 */

#ifndef __MEM_CACHE_PREFETCH_LSTM_HH__
#define __MEM_CACHE_PREFETCH_LSTM_HH__

#include <unordered_map>

#include "mem/cache/prefetch/queued.hh"
#include "params/LSTMPrefetcher.hh"
#include <vector>
#include "LSTM_ops.hh"
#include <algorithm>
#define  OFFSET_TABLE_SIZE 64
#define PAGE_TABLE_SIZE 64
#define INPUT_SIZE 13
#define NUM_BLOCKS 13
#define PAGE_OFFSET_MASK 63
#define DELTA_SHIFT 6
#define DELTA_MASK  63
#define PAGE_NUMBER_SHIFT 12
#define PAGE_NUMBER_MASK 255
#define PREV_ADDR_SHIFT 12
#define PREV_ADDR_MASK 4095
#define DEFAULT_DELTA 65
#define DEFAULT_PREVADDR -1
#define NUM_OF_DELTAS 4

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


//vars for the final address
    int curr_delta;
    int curr_page_num;
    int curr_offset;
    int curr_prevAddr;
    int final_prediction;
    int curr_page_table_idx;

    const double learning_rate;
    int prediction;
    double cell_state;
    double hidden_state;
    int block_input[INPUT_SIZE];

//backprop variables

    int back_block_input[INPUT_SIZE];
    int back_input[INPUT_SIZE];
    int back_forget[INPUT_SIZE];
    int back_output[INPUT_SIZE];



    struct offset_table_entry{
      int delta;
      int first_address;  //can also be obtained from the page table
      //state 0 means there is no match in the offset table (i.e. first access)
      //state 2 means that it hsa been accessed twice and holds a delta
      int state;
    };

    struct page_table_entry{

      int pageNumber; //7 bits
      int preAddr; //20 bits
      int pageSign //6 bits
      int prevDeltas[NUM_OF_DELTAS]; //7 bit twos complement
      int prev4Addrs[NUM_OF_DELTAS]; // 4 * 6 bits
      int accuracy; // 1 bit
      state_space stateSpace; // hidden and cell states
      state preState; // ^
    };

    struct state{
      double cell_state [INPUT_SIZE];
      int hidden_state [INPUT_SIZE];
    };
    struct forward_pass_args{
      double z_hat [INPUT_SIZE]; //block input
      double z [INPUT_SIZE];
      double i_hat [INPUT_SIZE]; //input gate
      double i [INPUT_SIZE];
      double f_hat  [INPUT_SIZE]; //forget
      double f [INPUT_SIZE];
      double c [INPUT_SIZE]; //cell state
      double o_hat [INPUT_SIZE]; //oyutput gate
      double o [INPUT_SIZE];
      double y [INPUT_SIZE]; //block output

    };
    struct state_space {
      forward_pass_args states[NUM_OF_DELTAS];
    };



    //pageoffset table
    //indexed by the page offset or bits [11:5]
    //A shared global array for first predictions, it holds the
    //first page offset and delta.

    offset_table_entry offset_table[OFFSET_TABLE_SIZE];

    //local page table for delta history for predictions.
    //also holds information for updating weights
    std::vector<page_table_entry> page_table;


    calculate_hidden(page_table_entry* entry);
    calculate_cell(page_table_entry* entry);
    update_params(page_table_entry* entry);
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
      int biasi, biasf, biaso, biasz;

    }
    //check "A Space Odyssey" for LSTM formulas

    //offset table and page table lookup on Prefetch activation event
    bool checkoffset(Addr pkt_addr);
    bool check_page_table();
    //equations and variable names taken from "LSTM: A Search Space Odyssey"
    void forward_prop(Addr pkt_addr);
    void backward_prop(Addr pkt_addr, int page_number);
    void evict_LRU(Addr pkt_addr);
    void update_offset_table(int offset, Addr pkt_addr);
    void new_page_entry();

  public:

    LSTMPrefetcher(const LSTMPrefetcherParams *p);

    void calculatePrefetch(const PacketPtr &pkt,
                           std::vector<AddrPriority> &addresses);
};

#endif // __MEM_CACHE_PREFETCH_LSTM_HH__
