
#include "LSTM_ops.hh"
#include <iostream>
#include <algorithm>
#include <vector>
#include <iterator>
using namespace std;
#define INPUT_SIZE 13
struct state{
  int cell_state [INPUT_SIZE];
  int hidden_state [INPUT_SIZE];
};
struct hi_test{
  int helloman;
};
vector<hi_test> hi_vec;
/*
void test_insert(int i){

  hi_test data = {{[0,0,0,0,0,0,0,0,0,0,0,0]}};
  hi_vec.insert(hi_vec.begin(), data);
}
*/
struct hi{
  int hello;
};

int main() {
    int test_array[NUM_BLOCKS][INPUT_SIZE];
    int test_array1[INPUT_SIZE];
    int test_array2[INPUT_SIZE];
    int test_array3[INPUT_SIZE];
    int testint;
    int testint1;


    /*
  std::vector<int>::iterator it;
    it = myvector.begin();
    it = myvector.insert ( it , hi );

  std::cout << "myvector contains:";
  for (it=myvector.begin(); it<myvector.end(); it++)
    std::cout << ' ' << *it;
  std::cout << '\n';
*/

//testing vector insert

/* VECTOR TESTING FOR PAGE TABLE REPLACEMENT
    vector<hi> myhellotest;
    myhellotest.push_back(hi());
    myhellotest
    cout << myhellotest.size();*/
    hi_test test_struct = {1};
  hi_vec.push_back(test_struct);

  hi_vec.insert(hi_vec.begin(),hi_vec[0]);
/*


    test_array[0][12] = 1;
    test_array1[12] = 1;
    matrix_product_forward_weights(test_array, test_array1, test_array2);
    for(int i = 0; i < INPUT_SIZE; i++){
      cout << test_array1[i];
    }
*/
/* matrix pointwise addition test

test_array2[0] = 1;
test_array1[0] = 1;
add_matrices(test_array1, test_array2, test_array3);
for(int i = 0; i < INPUT_SIZE; i++){
  cout << test_array3[i];
}


*/
/*
for(int i = 0; i < INPUT_SIZE; i++){
  test_array2[i] = 0;
}

test_array2[1] = 1;
num = from_array_to_num(test_array2);
cout << num;

*/
/*
test_array2[0] = 1;
test_array1[0] = 1;
pointwise_product(test_array1, test_array2, test_array3);
for(int i = 0; i < INPUT_SIZE; i++){
  cout << test_array3[i];
}

*/
// to bit array test
/*
    testint = -9;
    to_binary_array(testint, test_array1);
    for(int i = 0; i < INPUT_SIZE; i++){
      cout << test_array1[i];

    }
    cout <<endl;
    testint = from_array_to_num(test_array1);
    cout << testint;
*/
    return 0;

}
