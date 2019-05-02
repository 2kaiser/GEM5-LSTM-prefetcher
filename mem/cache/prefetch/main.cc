#define INPUT_SIZE 13
#include <iostream>
using namespace std;
#include <cmath>
#define LSB_MASK 1
#include <random>
#include <cstdlib>
#include <ctime>

inline int
from_array_to_num(int array[INPUT_SIZE]){
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
to_binary_array(int num, int array[INPUT_SIZE]){
  for(int i = 0; i < INPUT_SIZE; ++i){
    array[i] = (num >> i) & LSB_MASK;
  }


    return;

}

int main(){
/*

  int array[13];
  for(int i = 0; i<INPUT_SIZE;i++){
    array[i] = 0;

  }

  //array[1] = 1;
  array[12] = 1;
  //cout << from_array_to_num(array) << '\n';
  to_binary_array(-88,array);

  for(int i = 0; i<INPUT_SIZE;i++){
    cout << array[i] << '\n';

  }
  cout << from_array_to_num(array) << '\n';
*/
  double lower_bound = -.001;
  double upper_bound = .001;
  uniform_real_distribution<double> unif(lower_bound,upper_bound);
  default_random_engine re;
  for(int i = 0; i < 10; i++){
    printf("first %f second %f \n",unif(re), unif(re));

  }

  return 1;
}
