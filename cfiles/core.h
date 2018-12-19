#ifndef _CORE_H_
#define _CORE_H_

//最高维度个数，可更改
#define MAX_DIMENSION 5

typedef struct{
	float* data;
	int dims[MAX_DIMENSION];
	int max_dim;
}tensor;

void Conv2D(tensor* input_tensor,tensor* output_tensor,tensor* core);
void Reshape(tensor* t,int out_shape[],int dim);
void AddVector(tensor* t,tensor* v);
void ReLu(tensor* in,tensor* out);
void maxpool2x2(tensor* in,tensor* out);
void MatMul(tensor* t1,tensor* t2,tensor* output);
void softmax(tensor* input,tensor* output);
void tensortanh(tensor* in,tensor* out);
void sigmoid(tensor* in,tensor* out);

#endif