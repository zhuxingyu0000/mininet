#include "net.h"
#include "tensor_interface.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

int main()
{/**
	tensor v;
	v.data=(float*)malloc(sizeof(float)*28*28);
	v.max_dim=2;
	v.dims[0]=28;
	v.dims[1]=28;
	fstream f;
	f.open("arr.txt",ios::in);
	for(int i=0;i<28*28;i++)
		f>>v.data[i];
	f.close();
	tensor o;
	o.data=(float*)malloc(sizeof(float)*10);
	o.max_dim=1;
	o.dims[0]=10;
	run_net(&v,&o);
	tensor_print(&o);
	free(v.data);
	free(o.data);
	***/
	tensor v;
	float f[100];
	for(int i=1;i<=100;i++) f[i-1]=i;
	v.data=f;
	v.max_dim=4;
	v.dims[0]=1;
	v.dims[1]=10;
	v.dims[2]=10;
	v.dims[3]=1;
	maxpool2x2(&v,&v);
	tensor_print(&v);
	
	return 0;
}
