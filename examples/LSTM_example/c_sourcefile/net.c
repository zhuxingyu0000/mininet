#include "net.h"
#include "LSTM.h"

#include "weights/W1_C.h"
#include "weights/W1_I.h"
#include "weights/W1_F.h"
#include "weights/W1_O.h"
#include "weights/U1_C.h"
#include "weights/U1_I.h"
#include "weights/U1_F.h"
#include "weights/U1_O.h"
#include "weights/W_conv1.h"
#include "weights/b_conv1.h"

#include "tensor_interface.h"

#include "stdlib.h"

tensor W_conv1;
tensor b_conv1;
LSTM_cell global_LSTM_cell_1;

tensor t1;
tensor t2;

void global_net_init()
{
	LSTM_initalize_struct s_1={
		W1_I,
		U1_I,
		W1_C,
		U1_C,
		W1_F,
		U1_F,
		W1_O,
		U1_O
	};
	global_LSTM_cell_1=LSTM_cell_create(100,28);
	LSTM_cell_initalize(global_LSTM_cell_1,&s_1);

	W_conv1.dims[0]=10;
	W_conv1.dims[1]=100;
	W_conv1.max_dim=2;
	W_conv1.data=W_conv1_w;

	b_conv1.dims[0]=10;
	b_conv1.max_dim=1;
	b_conv1.data=b_conv1_w;

}

void run_net(tensor* in,tensor* out)
{
	int i;

	int temp_shape[MAX_DIMENSION];

	temp_shape[0]=28;
	temp_shape[1]=28;
	temp_shape[2]=-1;
	Reshape(in,temp_shape,3);
	
	t1.data=(float*)malloc(sizeof(float)*(100*in->dims[2]));
	LSTM_static(&global_LSTM_cell_1,1,1.0,in,&t1);
	
	t2.data=(float*)malloc(sizeof(float)*W_conv1.dims[0]*t1.dims[1]);
	MatMul(&t1,&W_conv1,&t2);
	free(t1.data);

	AddVector(&t2,&b_conv1,&t2);

	softmax(&t2,out);
	free(t2.data);
}

void global_net_destroy()
{
	LSTM_cell_destroy(global_LSTM_cell_1);
}
