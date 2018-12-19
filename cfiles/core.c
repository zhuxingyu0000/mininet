#include "core.h"
#include <assert.h>
#include <math.h>

#include <iostream>

using namespace std;

//取4维张量某行某列某维度上的值


inline float Tensor4(tensor* t,int x4,int x3,int x2,int x1)
{
	if(x1<0||x1>=t->dims[0]) return 0;
	if(x2<0||x2>=t->dims[1]) return 0;
	if(x3<0||x3>=t->dims[2]) return 0;
	if(x4<0||x4>=t->dims[3]) return 0;
	return t->data[x4*t->dims[2]*t->dims[1]*t->dims[0]+x3*t->dims[1]*t->dims[0]+x2*t->dims[0]+x1];
}

inline void Tensor4_Write(tensor* t,float n,int x4,int x3,int x2,int x1)
{
	if(x1<0||x1>=t->dims[0]) return;
	if(x2<0||x2>=t->dims[1]) return;
	if(x3<0||x3>=t->dims[2]) return;
	if(x4<0||x4>=t->dims[3]) return;
	t->data[x4*t->dims[2]*t->dims[1]*t->dims[0]+x3*t->dims[1]*t->dims[0]+x2*t->dims[0]+x1]=n;
}


// 求卷积，input_tensor被卷积量，output_tensor输出张量,core卷积核
void Conv2D(tensor* input_tensor,tensor* output_tensor,tensor* core)
{
	//input_tensor
	//dims[0] 输入图像深度
	//dims[1] dims[2] 输入图像x，y
	//dims[3] batch数量
	
	//core
	//dims[0] 输出图像深度
	//dims[1] 输入图像深度
	//dims[2] dims[3] 卷积核x，y
	
	int input_depth=input_tensor->dims[0];
	int input_x=input_tensor->dims[1];
	int input_y=input_tensor->dims[2];
	int batch=input_tensor->dims[3];
	
	int output_depth=core->dims[0];
	int core_x=core->dims[2];
	int core_y=core->dims[3];
	
	int i,d,x,y;
	
	//卷积核边长必须是奇数
	assert(core_x%2==1);
	assert(core_y%2==1);
	
	//计算输出张量尺寸
	output_tensor->dims[0]=core->dims[0];
	output_tensor->dims[1]=input_tensor->dims[1];
	output_tensor->dims[2]=input_tensor->dims[2];
	output_tensor->dims[3]=input_tensor->dims[3];
	output_tensor->max_dim=4;
	
	for(i=0;i<batch;i++)
	{
		for(d=0;d<output_depth;d++)
		{
			for(x=0;x<input_x;x++)
			{
				for(y=0;y<input_y;y++)
				{
					int j,m,n;
					int mid_x=core_x/2;//卷积核中心
					int mid_y=core_y/2;
					float sum=0.0;
					for(m=-mid_x;m<=mid_x;m++)
					{
						for(n=-mid_y;n<=mid_y;n++)
						{
							for(j=0;j<input_depth;j++)
							{
								sum+=Tensor4(input_tensor,i,y+n,x+m,j) *Tensor4(core,mid_y+n,mid_x+m,j,d);
							}
						}
					}
					Tensor4_Write(output_tensor,sum,i,y,x,d);
				}
			}
		}
	}
}

//类似numpy reshape函数
void Reshape(tensor* t,int out_shape[],int dim)
{
	int mul=1,mulmy=1;
	int i,m;
	for(i=0;i<dim;i++)
	{
		mul*=out_shape[i];
		if(out_shape[i]<0)
			m=i;
	}
	for(i=0;i<t->max_dim;i++)
		mulmy*=t->dims[i];
	if(mul>0) assert(mulmy==mul);
	else t->dims[m]=-mulmy/mul;
	for(i=0;i<dim;i++)
	{
		if(out_shape[i]>0)
			t->dims[i]=out_shape[i];
	}
	t->max_dim=dim;
}

// 将t与向量v相加
void AddVector(tensor* t,tensor* v)
{
	int mul=1,i,j;

	assert(t->dims[0]==v->dims[0]);
	assert(v->max_dim==1);
	
	for(i=1;i<t->max_dim;i++) mul*=t->dims[i];
	for(i=0;i<mul;i++)
	{
		for(j=0;j<v->dims[0];j++)
			t->data[i*t->dims[0]+j]+=v->data[j];
	}
}

inline float max_4(float x1,float x2,float x3,float x4)
{
	float y1=(x1>x2)?x1:x2;
	float y2=(x3>x4)?x3:x4;
	return (y1>y2)?y1:y2;
}

//池化
void maxpool2x2(tensor* in,tensor* out)
{
	int i,j,b,d;
	int p=0;
	//in
	//dims[0] 输入图像深度
	//dims[1] dims[2] 输入图像x，y
	//dims[3] batch数量
	//out->dims[0]=in->dims[0];
	//out->dims[1]=in->dims[1]/2;
	//out->dims[2]=in->dims[2]/2;
	//out->dims[3]=in->dims[3];
	//out->max_dim=4;
	for(b=0;b<in->dims[3];b++)
	{
		for(i=0;i<in->dims[2];i+=2)
		{
			for(j=0;j<in->dims[1];j+=2)
			{
				for(d=0;d<in->dims[0];d++)
				{
					float t1=Tensor4(in,b,i,j,d);
					float t2=Tensor4(in,b,i+1,j,d);
					float t3=Tensor4(in,b,i,j+1,d);
					float t4=Tensor4(in,b,i+1,j+1,d);
					float max=max_4(t1,t2,t3,t4);
					out->data[p++]=max;
				}
			}
		}
	}

	out->dims[0]=in->dims[0];
	out->dims[1]=in->dims[1]/2;
	out->dims[2]=in->dims[2]/2;
	out->dims[3]=in->dims[3];
	out->max_dim=4;
}

inline float Tensor2(tensor* t,int x2,int x1)
{
	if(x1<0||x1>=t->dims[0]) return 0;
	if(x2<0||x2>=t->dims[1]) return 0;
	return t->data[x2*t->dims[0]+x1];
}

inline void Tensor2_Write(tensor* t,float n,int x2,int x1)
{
	if(x1<0||x1>=t->dims[0]) return;
	if(x2<0||x2>=t->dims[1]) return;
	t->data[x2*t->dims[0]+x1]=n;
}


// 矩阵乘积
void MatMul(tensor* t1,tensor* t2,tensor* output)
{
	int i,j;
	assert(t1->max_dim<=2);
	assert(t2->max_dim<=2);
	//dims[0] 列数
	//dims[1] 行数
	//i行j列 Tensor2(t,i,j)
	if(t1->max_dim==1)
	{
		int newshape[]={-1,1};
		Reshape(t1,newshape,2);
	}
	if(t2->max_dim==1)
	{
		int newshape[]={-1,1};
		Reshape(t2,newshape,2);
	}
	assert(t1->dims[0]==t2->dims[1]);
	output->dims[0]=t2->dims[0];
	output->dims[1]=t1->dims[1];
	output->max_dim=2;
	for(i=0;i<output->dims[1];i++)
	{
		for(j=0;j<output->dims[0];j++)
		{
			float sum=0;
			int k;
			for(k=0;k<t1->dims[0];k++)
			{
				sum+=Tensor2(t1,i,k)*Tensor2(t2,k,j);
			}
			Tensor2_Write(output,sum,i,j);
		}
	}
}
