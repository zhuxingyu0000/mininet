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

#include "tensor_interface.h"

#include "stdlib.h"

LSTM_cell global_LSTM_cell_1;

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
	global_LSTM_cell_1=LSTM_cell_create(2,10);
	LSTM_cell_initalize(global_LSTM_cell_1,&s_1);
}

void run_net(tensor* in,tensor* out)
{
	int i;

	int temp_shape[MAX_DIMENSION];

	temp_shape[0]=10;
	temp_shape[1]=1;
	temp_shape[2]=-1;
	Reshape(in,temp_shape,3);
	
	LSTM_static(&global_LSTM_cell_1,1,in,out);

}

void global_net_destroy()
{
	LSTM_cell_destroy(global_LSTM_cell_1);
}
