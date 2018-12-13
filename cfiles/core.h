#ifndef _CORE_H_
#define _CORE_H_

#define MAX_DIMENSION 10

typedef struct{
	float* data;
	int dims[MAX_DIMENSION];
	int max_dim;
}tensor;

void Conv2D(tensor* input_tensor,tensor* output_tensor,tensor* core);

void tensor_print(tensor* t);

#endif