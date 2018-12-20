#ifndef _LSTM_H_
#define _LSTM_H_

#include "tensor.h"
#include "activation_function.h"
#include "basic.h"

typedef struct LSTM_cell_{
    int units;//LSTM输出序列长度
    int inputshape;//LSTM输入序列长度

    //多层LSTM用，单层LSTM设为本身
    struct LSTM_cell_* last;//前一LSTM单元
    struct LSTM_cell_* next;//后一LSTM单元

    //LSTM内部权值,W表示乘积矩阵，U表示偏置矩阵
    tensor W_I;
    tensor U_I;
    tensor W_C;
    tensor U_C;
    tensor W_F;
    tensor U_F;
    tensor W_O;
    tensor U_O;

    tensor C;//LSTM单元当前状态
    tensor L;//LSTM单元前一时刻状态

    //预留内存空间，保存LSTM单元计算结果
    tensor it,ct,ft,ot,input;
}*LSTM_cell;

typedef struct{
    float* W_I;
    float* U_I;
    float* W_C;
    float* U_C;
    float* W_F;
    float* U_F;
    float* W_O;
    float* U_O;
}LSTM_initalize_struct;

//LSTM接口

//生成LSTM单元（未初始化），units代表输出序列长度，inputshape代表输入序列长度
LSTM_cell LSTM_cell_create(int units,int inputshape);

//初始化LSTM单元的权值
void LSTM_cell_initalize(LSTM_cell cell,LSTM_initalize_struct* s);

//合并多层LSTM单元
void LSTM_multicellconnect(LSTM_cell* cell,int cells);

//运行静态LSTM,cell是存放LSTM单元的数组，cells代表多层LSTM的层数,time_steps代表LSTM的输入序列个数
//多层LSTM需要先合并
//LSTM input shape (batchs,time_steps,inputshape)
//LSTM output shape (batchs,outputshape)
void LSTM_static(LSTM_cell* cell,int cells,int forget_bias,tensor* input,tensor* output);

//销毁LSTM单元，释放alloc的内存
void LSTM_cell_destroy(LSTM_cell cell);

#endif