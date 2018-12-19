#ifndef _ACTIVATION_FUNCTION_H_
#define _ACTIVATION_FUNCTION_H_

#include "tensor.h"

void softmax(tensor* input,tensor* output);
void ReLu(tensor* in,tensor* out);
void tensortanh(tensor* in,tensor* out);
void sigmoid(tensor* in,tensor* out);

#endif
