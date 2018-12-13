#include "core.h"
#include "tensor_interface.h"
#include <stdlib.h>
#include <iostream>

using namespace std;

int main()
{
	tensor v;
	float d[]={0,0,0,0,-3,2,-1,0};
	v.data=d;
	v.max_dim=2;
	v.dims[0]=4;
	v.dims[1]=2;
	int shapenew[]={2,2,2};
	softmax(&v,&v);
	Reshape(&v,shapenew,3);
	tensor_print(&v);
	return 0;
}
