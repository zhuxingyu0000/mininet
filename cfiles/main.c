#include "core.h"
#include "tensor_interface.h"
#include <stdlib.h>

int main()
{
	tensor v;
	float d[]={-3,2,-1,0};
	v.data=d;
	v.max_dim=1;
	v.dims[0]=4;
	softmax(&v,&v);
	tensor_print(&v);
	return 0;
}
