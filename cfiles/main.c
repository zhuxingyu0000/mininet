#include "core.h"

#include <stdlib.h>

int main()
{
	tensor tensor1,tensor2,tensor3;

	tensor1.dims[0]=3;
	tensor1.dims[1]=3;
	tensor1.dims[2]=3;
	tensor1.dims[3]=1;
	tensor1.max_dim=4;
	float d1[]={100,100,100,100,100,100,100,100,100,
				100,100,100,100,100,100,100,100,100,
				100,100,100,100,100,100,100,100,100};
	tensor1.data=d1;
	tensor2.dims[0]=1;
	tensor2.dims[1]=3;
	tensor2.dims[2]=1;
	tensor2.dims[3]=1;
	tensor2.max_dim=4;
	float d2[]={0.5,0.5,0.5};
	
	tensor2.data=d2;
	tensor3.data=(float*)malloc(sizeof(float)*200);
	Conv2D(&tensor1,&tensor3,&tensor2);
	tensor_print(&tensor3);
	return 0;
}
