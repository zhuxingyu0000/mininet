#include "basic.h"

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
void AddVector(tensor* t,tensor* v,tensor* output)
{
	int mul=1,i,j;

	assert(t->dims[0]==v->dims[0]);
	assert(v->max_dim==1);
	
    output->max_dim=t->max_dim;

	for(i=1;i<t->max_dim;i++)
    {   
        mul*=t->dims[i];
        output->dims[i]=t->dims[i];
    }
	for(i=0;i<mul;i++)
	{
		for(j=0;j<v->dims[0];j++)
			output->data[i*t->dims[0]+j]+=v->data[j];
	}
}

