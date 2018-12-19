//该文件提供了常用的激活函数

#include "core.h"
#include <math.h>

//softmax函数
void softmax(tensor* input,tensor* output)
{
	int i,j;
	int batch_size=input->dims[0];
	int mul=1;
	output->max_dim=input->max_dim;
	for(i=0;i<output->max_dim;i++)
	{
		output->dims[i]=input->dims[i];
		mul*=input->dims[i];
	}
	mul/=input->dims[0];
	for(i=0;i<mul;i++)
	{
		float sum=0.0;
		for(j=0;j<batch_size;j++)
		{
			output->data[i*batch_size+j]=(float)exp((double)(input->data[i*batch_size+j]));
			sum+=output->data[i*batch_size+j];
		}
		for(j=0;j<batch_size;j++)
		{
			output->data[i*batch_size+j]=output->data[i*batch_size+j]/sum;
		}
	}
}

//ReLu激活函数
void ReLu(tensor* in,tensor* out)
{
	int i,mul=1;
	out->max_dim=in->max_dim;
	for(i=0;i<in->max_dim;i++)
	{
		out->dims[i]=in->dims[i];
		mul*=in->dims[i];
	}
	for(i=0;i<mul;i++)
	{
		out->data[i]=in->data[i];
		if(in->data[i]<0.0) out->data[i]=0.0;
	}
}

//tanh激活函数
void tensortanh(tensor* in,tensor* out)
{
    int i,mul=1;
	out->max_dim=in->max_dim;
	for(i=0;i<in->max_dim;i++)
	{
		out->dims[i]=in->dims[i];
		mul*=in->dims[i];
	}
	for(i=0;i<mul;i++)
	{
		out->data[i]=(float)tanh((double)in->data[i]);
	}
}

//sigmoid激活函数
void sigmoid(tensor* in,tensor* out)
{
    int i,mul=1;
	out->max_dim=in->max_dim;
	for(i=0;i<in->max_dim;i++)
	{
		out->dims[i]=in->dims[i];
		mul*=in->dims[i];
	}
	for(i=0;i<mul;i++)
	{
		out->data[i]=1/(1+(float)exp(-(double)in->data[i]));
	}
}
