#include "tensor_interface.h"
#include "core.h"

//该文件实现tensor类型的交互功能

#include <iostream>
using namespace std;

//打印张量t

void tensor_print_recursive(tensor* t,int m)
{
	cout<<'[';
	if(t->max_dim==1)
	{
		cout<<t->data[0];
		for(int i=1;i<t->dims[0];i++)
			cout<<' '<<t->data[i];
	}
	else
	{
		int k=1;
		for(int i=0;i<t->max_dim-1;i++) k*=t->dims[i];
		for(int i=0;i<t->dims[t->max_dim-1];i++)
		{
			tensor newtensor=*t;
			newtensor.max_dim=newtensor.max_dim-1;
			newtensor.data+=(i*k);
			tensor_print_recursive(&newtensor,m);
			cout<<endl;
			for(int j=0;j<m-t->max_dim+1;j++) cout<<' ';
		}
	}
	cout<<']';
}

void tensor_print(tensor* t)
{
	tensor_print_recursive(t,t->max_dim);
	cout<<endl;
}
