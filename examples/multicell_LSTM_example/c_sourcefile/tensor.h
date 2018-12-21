#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <assert.h>
#include <iostream>
using namespace std;

//最高维度个数，可更改
#define MAX_DIMENSION 5

typedef struct{
	float* data;
	int dims[MAX_DIMENSION];
	int max_dim;
}tensor;

#endif
