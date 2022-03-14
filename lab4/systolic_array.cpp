#include "systolic_array.h"
//#include <cmath>

template<typename T>
struct Process_Element
{
	T a, b, val;

	Process_Element()
	{
		reset();
	}

	void reset()
	{
		a = 0;
		b = 0;
		val = 0;
	}

	void process(T a_i, T b_i)
	{
		a = a_i;			// a�������Ҵ���
		b = b_i;			// b�������´���
		val += a_i*b_i;
	}
};

template<typename T, int LEN>
struct Systolic_Array
{
	// ����������PE��ɵ��������з���
	Process_Element<T> pe[LEN][LEN];

	void reset()
	{
		for (int r = 0; r < LEN; r++)
			for (int c = 0; c < LEN; c++)
				pe[r][c].reset();
	}

	void reset(int row, int col)
	{
		for (int r = 0; r < row; r++)
			for (int c = 0; c < col; c++)
				pe[r][c].reset();
	}

	// ִ��һ��pulse�������������������е�ˮƽ�������ֱ�����Ͼ�����һ�������������е���������һ��
	void pulse(T a_vec[LEN], T b_vec[LEN])
	{
		systolic_array_outer_loop:
		// ��������Ͻ��ߵ����½ǣ��ܹ���Ҫ2*LEN-1��step�����ž���ıߴ�һ���Խ��ߵ���һ���Խ�
		for (int i = 2*LEN - 2; i >= 0; i--)
		{
#pragma HLS UNROLL
			// TODO: ����ÿһ��step��Ҫ�����PE��
			int pe_num = (i > LEN - 1)?(2 * LEN - 1 - i):(i + 1);

			systolic_array_inner_loop:
			for (int j = 0; j < pe_num; j++)
			{
//#pragma HLS UNROLL factor=10
				// TODO: ��ȡ��ǰPE������
				int pos_y = (i >= LEN)?LEN-1-j:i-j;
				int pos_x = (i >= LEN)?i-LEN+j+1:j;

				// TODO: ��ȡ��ǰPE���������a���Ϸ�����b
				T a_get = (pos_y == 0)?a_vec[pos_x]:pe[pos_x][pos_y-1].a;
				T b_get = (pos_x == 0)?b_vec[pos_y]:pe[pos_x-1][pos_y].b;
				// TODO: ����a��b����PE
				pe[pos_x][pos_y].process(a_get, b_get);
//				pe[pos_x][pos_y].val+=(a_get*b_get);
//				pe[pos_x][pos_y].a = a_get;
//				pe[pos_x][pos_y].b = b_get;
			}
		}
	}

	// ִ��һ��pulse��������Ч�������������е�2�������Ͼ�����һ��������������PE����һ�ģ�����PE����
	void pulse(T a_vec[LEN], int a_size, T b_vec[LEN], int b_size)
	{
		int shorter, longer;
		if (a_size < b_size)
		{
			shorter = a_size;
			longer  = b_size;
		}
		else
		{
			shorter = b_size;
			longer  = a_size;
		}

		systolic_array_outer_loop:
		// ���д����Ͻ��ߵ����½ǣ��ܹ���Ҫ???��step
		for (int i = shorter + longer - 2; i >= 0; i--)
		{
			// TODO: ����ÿһ��step��Ҫ�����PE��
			int pe_num = (i > longer - 1)?shorter+longer-1-i:
						(i > shorter - 1)?shorter:i+1;

			systolic_array_inner_loop:
			for (int j = 0; j < pe_num; j++)
			{
//#pragma HLS PIPELINE II=1
				// TODO: ��ȡ��ǰPE������
				int pos_y = (i >= b_size)?b_size - j - 1:i-j;
				int pos_x = (i >= b_size)?i-b_size+1+j:j;
				// TODO: ��ȡ��ǰPE���������a���Ϸ�����b
				T a_get = (pos_y == 0)?a_vec[pos_x]:pe[pos_x][pos_y-1].a;
				T b_get = (pos_x == 0)?b_vec[pos_y]:pe[pos_x-1][pos_y].b;
				// TODO: ����a��b����PE
				pe[pos_x][pos_y].val+=(a_get*b_get);
				pe[pos_x][pos_y].a = a_get;
				pe[pos_x][pos_y].b = b_get;
			}
		}
	}

//  Debug
//	void print_pe(int len, int width)
//	{
//		printf("Pe now is below:\n");
//		for(int i = 0; i < len; i++)
//		{
//			for(int j = 0; j < width; j++)
//			{
//				printf("(%f, %f, %f)\t", pe[i][j].a, pe[i][j].b, pe[i][j].val);
//			}
//			printf("\n");
//		}
//	}
};

// �������б���
Systolic_Array<DataType, SIDE_LEN> systolic_matrix;

void gemm_kernel(int piece_a_cell, int piece_b_cell, int row, int col, int col1, int ori_col1, DataType din_a[], DataType din_b[])
{
	systolic_matrix.reset(row, col1);

	// ��������ˮƽ�����������������ֱ�������������
	DataType a_vec[SIDE_LEN], b_vec[SIDE_LEN];

	// TODO: �����������м������ʱ��������Ҫ�����Ĵ���
	int total_pulse = col + row + col1 -2;

	int bigger = (row>col1)?row:col1;

	gemm_outer_loop:
	for (int i = 0; i < total_pulse; i++)
	{
		gemm_inner_loop:
		for (int j = 0; j < bigger; j++)
		{
			int a_index = piece_a_cell*col*SIDE_LEN + j*col + i - j;
			int b_index = (i - j)*ori_col1 + j + piece_b_cell*SIDE_LEN;
			// TODO: ��һ��ȡ�����������������ĸ���Ԫ��
			a_vec[j] = (i >= j && i < j+col && j < row) ? din_a[a_index] : 0;
			b_vec[j] = (i >= j && i < j+col && j < col1) ? din_b[b_index] : 0;
		}
		systolic_matrix.pulse(a_vec, row, b_vec, col1);
	}
}

// ���������м����������������
void copy_result(int piece_a_cell, int piece_b_cell, int row, int col1, int ori_col1, DataType bias[], DataType out[])
{
	// TODO
	for(int i = 0; i < row; i++)
	{
#pragma HLS UNROLL factor=30
		for(int j = 0; j < col1; j++)
//#pragma HLS loop_flatten
			out[piece_a_cell*SIDE_LEN*ori_col1+i*ori_col1+(piece_b_cell*SIDE_LEN+j)]
				= systolic_matrix.pe[i][j].val
					+ bias[piece_a_cell*SIDE_LEN+i];
	}


}

// THIS IS THE TOP LEVEL DESIGN THAT WILL BE SYNTHESIZED (Size-Free version)
void systolic_array(ap_uint<16> row, ap_uint<16> col, ap_uint<16> col1, DataType din_a[], DataType din_b[], DataType bias[], DataType out[])
{
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=row
#pragma HLS INTERFACE s_axilite port=col
#pragma HLS INTERFACE s_axilite port=col1
#pragma HLS INTERFACE m_axi depth=4294967295 port=din_a offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=din_b offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=bias  offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=out   offset=slave

//	int piece_a = (int)ceil((float)row/SIDE_LEN);
//	int piece_b = (int)ceil((float)col1/SIDE_LEN);
	int piece_a = row/SIDE_LEN + (row%SIDE_LEN > 0);		// ����˷��������������Ҫ�����гɼ���
	int piece_b = col1/SIDE_LEN + (col1%SIDE_LEN > 0);		// ����˷��Ҳ�����������Ҫ�����гɼ���
	int piece_a_row, piece_b_col, piece_a_cell, piece_b_cell;

#pragma HLS DATAFLOW
	top_outer_loop:
	for (int i = 0; i < piece_a; i++)
	{
		// Get piece index of array a
		piece_a_cell = i;
		// TODO: ���㵱ǰ�ֿ������
		piece_a_row = (i==piece_a-1)?(int)(row-piece_a_cell*SIDE_LEN):SIDE_LEN;

		top_inner_loop:
		for (int j = 0; j < piece_b; j++)
		{
			// Get piece index of array b
			piece_b_cell = j;
			// TODO: ���㵱ǰ�ֿ������
			piece_b_col = (j==piece_b-1)?(int)(col1-piece_b_cell*SIDE_LEN):SIDE_LEN;
			// Using gemm kernel to perform matrix multiplication
			gemm_kernel(piece_a_cell, piece_b_cell, piece_a_row, col, piece_b_col, col1, din_a, din_b);
			// Copy gemm result to output port
			copy_result(piece_a_cell, piece_b_cell, piece_a_row, piece_b_col, col1, bias, out);
		}
	}
}

//// THIS IS THE TOP LEVEL DESIGN THAT WILL BE SYNTHESIZED (Size-Limited version)
//void systolic_array(ap_uint<16> row, ap_uint<16> col, ap_uint<16> col1, DataType din_a[], DataType din_b[], DataType bias[], DataType out[])
//{
//#pragma HLS INTERFACE s_axilite port=return
//#pragma HLS INTERFACE s_axilite port=row
//#pragma HLS INTERFACE s_axilite port=col
//#pragma HLS INTERFACE s_axilite port=col1
//#pragma HLS INTERFACE m_axi depth=4294967295 port=din_a offset=slave
//#pragma HLS INTERFACE m_axi depth=4294967295 port=din_b offset=slave
//#pragma HLS INTERFACE m_axi depth=4294967295 port=bias  offset=slave
//#pragma HLS INTERFACE m_axi depth=4294967295 port=out   offset=slave
//
//	systolic_matrix.reset();
//
//	// ��������ˮƽ�����������������ֱ�������������
//	DataType a_vec[SIDE_LEN], b_vec[SIDE_LEN];
//
//	// TODO: �����������м������ʱ��������Ҫ�����Ĵ���
//	int total_pulse = col + row + col1 -2;
//
//	top_outer_loop1:
//	for (int i = 0; i < total_pulse; i++)
//	{
//		top_inner_loop1:
//		for (int j = 0; j < SIDE_LEN; j++)
//		{
//			int a_index = j*col + i - j;
//			int	b_index = (i - j)*col1 + j;
//
//			// TODO: ��һ��ȡ�����������������ĸ���Ԫ��
////			a_vec[j] = (???) ?din_a[a_index] : 0;
////			b_vec[j] = (???) ? din_b[b_index] : 0;
//			a_vec[j] = (i >= j && i < j+col && i < row) ? din_a[a_index] : 0;
//			b_vec[j] = (i >= j && i < j+col && i < col1) ? din_b[b_index] : 0;
//		}
//
//		// ��������
//		systolic_matrix.pulse(a_vec, b_vec);
//	}
//
//	// TODO: ���������м����������������
//	int ROW = ((int)row < SIDE_LEN) ? (int)row : SIDE_LEN;
//	int COL = ((int)col1 < SIDE_LEN) ? (int)col1 : SIDE_LEN;
//
//	for(int i = 0; i < row; i++)
//	{
//		for(int j = 0; j < col1; j++)
//			out[i*SIDE_LEN+j] = systolic_matrix.pe[i][j].val+ bias[i*SIDE_LEN+j];
//	}
//}
