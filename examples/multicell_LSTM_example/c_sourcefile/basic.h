#ifndef _BASIC_H_
#define _BASIC_H_

#include "tensor.h"

void Reshape(tensor* t,int out_shape[],int dim);
void AddVector(tensor* in,tensor* v,tensor* out);
void MatMul(tensor* t1,tensor* t2,tensor* output);

#endif
