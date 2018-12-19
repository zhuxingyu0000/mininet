#ifndef _CNN_H_
#define _CNN_H_

#include "tensor.h"

void Conv2D(tensor* input_tensor,tensor* output_tensor,tensor* core);
void maxpool2x2(tensor* in,tensor* out);

#endif
