#include "net.h"
#include "tensor_interface.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

int main()
{
	tensor v;
	v.data=(float*)malloc(sizeof(float)*28*28);
	v.max_dim=2;
	v.dims[1]=28;
	v.dims[0]=28;
	fstream f;
	f.open("arr.txt",ios::in);
	for(int i=0;i<28*28;i++)
		f>>v.data[i];
	f.close();
	tensor o;
	o.data=(float*)malloc(sizeof(float)*10);
	o.max_dim=1;
	o.dims[0]=10;
	global_net_init();
	run_net(&v,&o);
	tensor_print(&o);
	global_net_destroy();
	free(v.data);
	free(o.data);
	return 0;
}
