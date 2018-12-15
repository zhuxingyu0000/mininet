#include "net.h"
#include "weights/W1.h"
#include "weights/b1.h"
#include "weights/W2.h"
#include "weights/b2.h"
#include "weights/W_fc1.h"
#include <stdlib.h>

void net_run(tensor* in,tensor* out)
{
	int i;

	int temp_shape[MAX_DIMENSION];
	tensor W1;
	tensor b1;
	tensor W2;
	tensor b2;
	tensor W_fc1;
	tensor b_fc1;

	tensor t1;
	tensor t2;
	tensor t3;
	tensor t4;

	temp_shape[0]=1;
	temp_shape[1]=28;
	temp_shape[2]=28;
	temp_shape[3]=-1;
	Reshape(in,temp_shape,4);
	
	W1.dims[0]=32;
	W1.dims[1]=1;
	W1.dims[2]=5;
	W1.dims[3]=5;
	W1.max_dim=4;
	W1.data=W1_w;
	t1.data=(float*)malloc(sizeof(float)*(32*in->dims[1]*in->dims[2]*in->dims[3]]));
	Conv2D(in,&t1,&W1);
	
	b1.dims[0]=32;
	b1.max_dim=1;
	b1.data=b1_w;
	AddVector(&t1,&b1);
	
	ReLu(&t1,&t1);
	
	maxpool2x2(&t1,&t1);
	
	W2.dims[0]=64;
	W2.dims[1]=32;
	W2.dims[2]=5;
	W2.dims[3]=5;
	W2.max_dim=4;
	W2.data=W2_w;
	t2.data=(float*)malloc(sizeof(float)*(64*t1->dims[1]*t1->dims[2]*t1->dims[3]]));
	Conv2D(&t1,&t2,&W2);
	free(t1.data);
	
	b2.dims[0]=64;
	b2.max_dim=1;
	b2.data=b2_w;
	AddVector(&t2,&b2);
	
	ReLu(&t2,&t2);
	
	maxpool2x2(&t2,&t2);
	
	temp_shape[0]=3136;
	temp_shape[1]=-1;
	Reshape(&t2,temp_shape,2);

	W_fc1.dims[0]=1024;
	W_fc1.dims[1]=3136;
	W_fc1.max_dim=2;
	W_fc1.data=W_fc1_w;
	t3.data=(float*)malloc(sizeof(float)*W_fc1.dims[0]*t2.dims[1]);
	MatMul(&t2,&W_fc1,&t3);
	free(t2.data);

	b_fc1.dims[0]=1024;
	b_fc1.max_dim=1;
	b_fc1.data=b_fc1_w;
	AddVector(&t3,&b_fc1);

	ReLu(&t3,&t3);

	W_fc2.dims[0]=10;
	W_fc2.dims[1]=1024;
	W_fc2.max_dim=2;
	W_fc2.data=W_fc2_w;
	t6.data=(float*)malloc(sizeof(float)*W_fc2.dims[0]*t3.dims[1]);
	MatMul(&t5,&W_fc2,&t4);
	free(t3.data);

	b_fc2.dims[0]=10;
	b_fc2.max_dim=1;
	b_fc2.data=b_fc2_w;
	AddVector(&t4,&b_fc2);

	softmax(&t4,out);
	free(t4.data);
}
