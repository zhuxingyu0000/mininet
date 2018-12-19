#include "core.h"
#include "RNN.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

LSTM_cell LSTM_cell_create(int units,int inputshape)
{
    LSTM_cell cell=(LSTM_cell)malloc(sizeof(struct LSTM_cell_));
    cell->units=units;
    cell->inputshape=inputshape;
    cell->last=cell;
    cell->next=cell;
    
    cell->W_I.max_dim=2;
    cell->W_I.dims[1]=inputshape+units;
    cell->W_I.dims[0]=units;

    cell->U_I.max_dim=1;
    cell->U_I.dims[0]=units;

    cell->W_C.max_dim=2;
    cell->W_C.dims[1]=inputshape+units;
    cell->W_C.dims[0]=units;

    cell->U_C.max_dim=1;
    cell->U_C.dims[0]=units;

    cell->W_F.max_dim=2;
    cell->W_F.dims[1]=inputshape+units;
    cell->W_F.dims[0]=units;

    cell->U_F.max_dim=1;
    cell->U_F.dims[0]=units;

    cell->W_O.max_dim=2;
    cell->W_O.dims[1]=inputshape+units;
    cell->W_O.dims[0]=units;

    cell->U_O.max_dim=1;
    cell->U_O.dims[0]=units;

    cell->C.data=(float*)malloc(sizeof(float)*units);
    cell->C.max_dim=1;
    cell->C.dims[0]=units;

    cell->L.data=(float*)malloc(sizeof(float)*units);
    cell->L.max_dim=1;
    cell->L.dims[0]=units;

    //预留内存空间，计算时要用
    cell->it.data=(float*)malloc(sizeof(float)*units);
    cell->ct.data=(float*)malloc(sizeof(float)*units);
    cell->ft.data=(float*)malloc(sizeof(float)*units);
    cell->ot.data=(float*)malloc(sizeof(float)*units);
    cell->input.data=(float*)malloc(sizeof(float)*(units+inputshape));

    cell->input.max_dim=2;
    cell->input.dims[0]=inputshape+units;
    cell->input.dims[1]=1;

    return cell;
}

void LSTM_cell_initalize(LSTM_cell cell,LSTM_initalize_struct* s)
{
    cell->U_I.data=s->U_I;
    cell->U_C.data=s->U_C;
    cell->U_F.data=s->U_F;
    cell->U_O.data=s->U_O;
    cell->W_I.data=s->W_I;
    cell->W_C.data=s->W_C;
    cell->W_F.data=s->W_F;
    cell->W_O.data=s->W_O;
}

void LSTM_multicellconnect(LSTM_cell* cell,int cells)
{
    int i;
    for(i=1;i<cells-1;i++)
    {
        cell[i]->next=cell[i+1];
        cell[i]->last=cell[i-1];
    }
    cell[0]->last=cell[0];
    cell[cells-1]->next=cell[cells-1];
}

void LSTM_cell_destroy(LSTM_cell cell)
{
    free(cell->C.data);
    free(cell->L.data);
    free(cell->it.data);
    free(cell->ct.data);
    free(cell->ot.data);
    free(cell->ft.data);
    free(cell);
}

//执行LSTM单元，input为输入一维数组
void LSTM_call(LSTM_cell cell,float* input)
{
    int i;

    //input=[L(t-1),x(t)]
    for(i=0;i<cell->units;i++) cell->input.data[i]=cell->last->L.data[i];
    for(i=0;i<cell->inputshape;i++) cell->input.data[i+cell->units]=input[i];

    //遗忘门
    //f(t)=sigma(Wf*input+bf)
    MatMul(&(cell->input),&(cell->W_F),&(cell->ft));
    AddVector(&(cell->ft),&(cell->U_F));
    sigmoid(&(cell->ft),&(cell->ft));

    //输入门
    //i(t)=sigma(Wi*input+bi)
    //C~(t)=tanh(Wc*input+bc)
    MatMul(&(cell->input),&(cell->W_I),&(cell->it));
    AddVector(&(cell->it),&(cell->U_I));
    sigmoid(&(cell->it),&(cell->it));

    MatMul(&(cell->input),&(cell->W_C),&(cell->ct));
    AddVector(&(cell->ct),&(cell->U_C));
    tensortanh(&(cell->ct),&(cell->ct));

    //输出门
    //o(t)=sigma(Wo*input+bo)
    MatMul(&(cell->input),&(cell->W_O),&(cell->ot));
    AddVector(&(cell->ot),&(cell->U_O));
    sigmoid(&(cell->ot),&(cell->ot));

    //更新C(t)
    //C(t)=f(t)*C(t-1)+i(t)*C~(t)
    for(i=0;i<cell->units;i++) cell->C.data[i]=cell->ft.data[i]*cell->C.data[i]+cell->it.data[i]*cell->ct.data[i];

    //更新输出状态
    //L(t)=o(t)*tanh(C(t))
    tensortanh(&(cell->ct),&(cell->L));
    for(i=0;i<cell->units;i++) cell->L.data[i]=cell->ot.data[i]*cell->L.data[i];
}

void LSTM_static(LSTM_cell* cell,int cells,tensor* input,tensor* output)
{
    int i,j,batchs,time_steps;

    time_steps=input->dims[1];
    batchs=input->dims[2];

    //输入维度个数，要么是2（一个batch）,要么是3
    assert(input->max_dim<4);
    if(input->max_dim==2)//只有一个batch的情况
    {
        int shape[3]={input->dims[0],input->dims[1],1};
        Reshape(input,shape[3],3);
    }
    assert(input->dims[0]==cell[0]->inputshape);

    output->dims[0]=cell[0]->units;
    output->dims[1]=batchs;
    output->max_dim=2;

    for(i=0;i<batchs;i++)
    {
        int k;
        for(j=0;j<cells;j++)
        {
            //清空cell的状态和上一状态值
            memset(cell[i]->C.data,0,sizeof(float)*cell[i]->units);
            memset(cell[i]->L.data,0,sizeof(float)*cell[i]->units);
        }
        for(j=0;j<time_steps;j++)
        {
            float* data=input->data+input->dims[0]*j+input->dims[0]*input->dims[1]*i;
            for(k=0;k<cells;k++)
            {
                LSTM_call(cell[k],data);
            }
        }
        for(k=0;k<output->dims[0];k++)
        {
            output->data[k+output->dims[0]*i]=cell[cells-1]->L.data[k];
        }
    }
}
