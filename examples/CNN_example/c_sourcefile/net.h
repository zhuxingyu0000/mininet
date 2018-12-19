#ifndef _NET_H_
#define _NET_H_

#include "tensor.h"
#include "basic.h"
#include "activation_function.h"

void run_net(tensor* in,tensor* out);
void global_net_init();
void global_net_destroy();

#endif