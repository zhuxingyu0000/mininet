#include "net.h"
#include "tensor_interface.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

int main()
{
	tensor v;
	v.data=(float*)malloc(sizeof(float)*10);
	v.max_dim=1;
	v.dims[0]=10;
	fstream f;
	f.open("arr.txt",ios::in);
	for(int i=0;i<10;i++)
		f>>v.data[i];
	f.close();
	tensor o;
	o.data=(float*)malloc(sizeof(float)*2);
	o.max_dim=1;
	o.dims[0]=2;
	global_net_init();
	run_net(&v,&o);
	tensor_print(&o);
	global_net_destroy();
	free(v.data);
	free(o.data);
	return 0;
}
