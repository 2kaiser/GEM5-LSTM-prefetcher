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

#include "base/random.hh"
#include "base/trace.hh"
#include "debug/HWPrefetch.hh"
#include "LSTM_ops.hh"



LSTMPrefetcher::LSTMPrefetcher(const LSTMPrefetcherParams *p)
    : QueuedPrefetcher(p)
          {
    // Don't consult LSTM prefetcher on instruction accesses
    onInst = false;
    //from thesis paper
    learning_rate = .2;
    double init_weights_low = -.0001;
    double init_weights_hi = .0001;
    weight_size = sizeOf(weighto_r[0]);


    //initializing the LSTM parameters
    //initialze weights to random value between -10^-5 and 10^-5
    //from LSTM master thesis paper
    for(int i = 0; i< weight_size; i++){
      for(int j =0; j < weight_size; i++){
        weighti_i[i][j] = random_mt.random(init_weights_low, init_weights_hi);
        weighti_r[i][j] = random_mt.random(init_weights_low, init_weights_hi);
        weightf_i[i][j] = random_mt.random(init_weights_low, init_weights_hi);
        weightf_r[i][j] = random_mt.random(init_weights_low, init_weights_hi);
        weighto_i[i][j] = random_mt.random(init_weights_low, init_weights_hi);
        weighto_r[i][j] = random_mt.random(init_weights_low, init_weights_hi);
        weightg_i[i][j] = random_mt.random(init_weights_low, init_weights_hi);
        weightg_r[i][j] = random_mt.random(init_weights_low, init_weights_hi);
      }
    }
    //refer to An Empirical Exploration of Recurrent Network Architectures (Jozefowicz et al., 2015): for bias initialization

     biasi  = 1;
     biasf = 1;
     biaso = 1;
     biasg = 1;

    //initialze offset table states to 0
    for(int i = 0; i < OFFSET_TABLE_SIZE; i++){
      offset_table[i].state = 0;
    }
}
}
inline void
LSTMPrefetcher::backward_prop(Addr pkt_addr_correct, int page_number){
  /*
  add 0 to dy for the 5th timestep since there are no future time steps
  */



}

inline void
LSTMPrefetcher::forward_prop(Addr pkt_addr){

//don't forget to change cache line size and use the makeline function? prob dont have to
  //use prestate for initial hidden state
  //index into the page table.  use the
  //form the input [x,y] where y is the hidden statefrom the previous timestep
  int x_t[INPUT_SIZE];
  double temp[INPUT_SIZE];

  for(int i = 0; i < NUM_OF_DELTAS + 1; i++){
    if(i ==4){
      to_binary_array(curr_delta,x_t);


    }
    else{
      to_binary_array(page_table_entry[curr_page_table_idx].prevDeltas[i],x);
      //z_hat calculation
      matrix_product_forward_weights(weightz_i, x,page_table[curr_page_table_idx].stateSpace[i].z_hat);
      matrix_product_forward_weights(weightz_r, page_table[curr_page_table_idx].stateSpace[i].y,temp);
      add_matrices(page_table[curr_page_table_idx].stateSpace[i].z_hat,temp,page_table[curr_page_table_idx].stateSpace[i].z_hat);
      add_bias(page_table[curr_page_table_idx].stateSpace[i].z_hat, biasz);
      //add in gating
    }
  }



}
inline void
LSTMPrefetcher::new_page_entry(){

  page_table_entry new_entry = {
    curr_page_num,
    curr_prevAddr,
    0,
    {DEFAULT_DELT, DEFAULT_DELTA, DEFAULT_DELTA, DEFAULT_DELTA}, //
    {DEFAULT_PREVADDR, DEFAULT_PREVADDR, DEFAULT_PREVADDR, DEFAULT_PREVADDR}, //so addresses don't match for prediction
    0,
    {
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
      {0,0,0,0,0,0,0,0,0,0,0,0,0}{
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
      page_table = i;
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
      if(page_tabl[curr_page_table_idx].prevDeltas[i] == DEFAULT_DELTA){
        delta_count++;
      }
      if(page_tabl[curr_page_table_idx].prev4Addrs[i] == curr_offset){
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
    offset_table[curr_offset].delta = (offset_table[offset_index].first_address >> DELTA_SHIFT) & DELTA_MASK - ((pkt_addr >> DELTA_SHIFT) & DELTA_MASK);
    curr_delta = offset_table[curr_offset].delta;
    return 0; //no prediction
  }
  else{
    curr_delta = offset_table[curr_offset].delta;

    return 1; //no prediction

  }
}

void
LSTMPrefetcher::calculatePrefetch(const PacketPtr &pkt,
                                    std::vector<AddrPriority> &addresses)
{
    if (!pkt->req->hasPC()) {
        DPRINTF(HWPrefetch, "Ignoring request with no PC.\n");
        return;
    }
    //populate current variables used in prediction

    int curr_offset = (pkt_addr & PAGE_OFFSET_MASK);
    int curr_page_num = (pkt_addr >> PAGE_NUMBER_SHIFT) & PAGE_NUMBER_MASK;
    int curr_prevAddr = (pkt_addr >> PREV_ADDR_SHIFT) & PREV_ADDR_MASK;

    // Get required packet info
    Addr pkt_addr = pkt->getAddr();
    Addr pc = pkt->req->getPC();
    bool is_secure = pkt->isSecure();
    MasterID master_id = useMasterId ? pkt->req->masterId() : 0;

    if (checkoffset(pkt_addr)) {
        //evict or update page table entry
        if(check_page_table(pkt_addr)){
          //found a offset hit in prev4fetchAddrs and do a prediction with the four deltas
          //update weights before prediction
            backward_prop();
            final_prediction = forward_prop();

            Addr new_addr = (curr_page_num << PAGE_NUMBER_SHIFT) + curr_offset + curr_delta << DELTA_SHIFT;
            //update page_table
            int idx;
            for(int i = 0; i < NUM_OF_DELTAS-1; i++){
            idx = 3-i;
            page_table[curr_page_table_idx].prevDeltas[idx] = page_table[curr_page_table_idx].prevDeltas[idx-1];
            page_table[curr_page_table_idx].prev4Addrs[idx] = page_table[curr_page_table_idx].prevDeltas[idx-1];

          }
          //update the first entry
          page_table[curr_page_table_idx].prevDeltas[0] = curr_delta;
          page_table[curr_page_table_idx].prev4Addrs[0] = curr_offset;

            if (samePage(pkt_addr, new_addr)) {
                DPRINTF(HWPrefetch, "Queuing prefetch to %#x.\n", new_addr);
                addresses.push_back(AddrPriority(new_addr, 0));
            } else {
                // Record the number of page crossing prefetches generated
                pfSpanPage += degree - d + 1;
                DPRINTF(HWPrefetch, "Ignoring page crossing prefetch.\n");
                return;
            }

          }
          else{  //first access to a page so use the delta in the offset table
            // Miss in page table
            //push on addr using delta from offset table
            Addr new_addr = (curr_page_num << PAGE_NUMBER_SHIFT) + curr_offset + curr_delta << DELTA_SHIFT;
            //update page_table
            page_table[curr_page_table_idx].prevDeltas[0] = curr_delta;
            page_table[curr_page_table_idx].prev4Addrs[0] = curr_offset;
            if (samePage(pkt_addr, new_addr)) {
                DPRINTF(HWPrefetch, "Queuing prefetch to %#x.\n", new_addr);
                addresses.push_back(AddrPriority(new_addr, 0));
            } else {
                // Record the number of page crossing prefetches generated
                pfSpanPage += degree - d + 1;
                DPRINTF(HWPrefetch, "Ignoring page crossing prefetch.\n");
                return;
            }


          }

    } else {

      //miss in offset table
      //updated offest table in previous call
      //update page_table
      bool ignore = check_page_table(); //this will insert new entry if necessary
      return;
    }
}



LSTMPrefetcher*
LSTMPrefetcherParams::create()
{
    return new LSTMPrefetcher(this);
}
