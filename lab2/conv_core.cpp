#include "conv_core.h"

//Feature: [H][W][C]
//kernel: [Ky][Kx][CHin][CHout]

/* Original Signature
void Conv(ap_uint<16> CHin,ap_uint<16> Hin,ap_uint<16> Win,ap_uint<16> CHout,
		ap_uint<8> Kx,ap_uint<8> Ky,ap_uint<8> Sx,ap_uint<8> Sy,ap_uint<1> mode,ap_uint<1> relu_en,
		Dtype_f feature_in[],Dtype_w W[],Dtype_w bias[],Dtype_f feature_out[]
	)//mode: 0:VALID, 1:SAME
*/
// New Version
void Conv(ap_uint<16> CHin,ap_uint<16> Hin,ap_uint<16> Win,ap_uint<16> CHout,
		ap_uint<8> Kx,ap_uint<8> Ky,ap_uint<8> Sx,ap_uint<8> Sy,ap_uint<1> mode,ap_uint<1> relu_en,
		ap_int<8> feature_in[],ap_int<8> W[],Dtype_w bias[],Dtype_f feature_out[],Dtype_f qmax[2]
	)//mode: 0:VALID, 1:SAME
{
	//#pragma HLS PIPELINE enable_flush
	#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_out offset=slave
	#pragma HLS INTERFACE m_axi depth=4294967295 port=bias offset=slave
	#pragma HLS INTERFACE m_axi depth=4294967295 port=W offset=slave
	#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_in offset=slave
	/* 提示: 为参数qmax添加总线接口信号定义, 即Directive */
	// TODO:
//	#pragma HLS INTERFACE s_axilite port=qmax   // 使用axi_lite
//	#pragma HLS INTERFACE m_axi depth=4294967295 port=qmax offset=slave  // 使用axi_lite
	#pragma HLS INTERFACE m_axi depth=4294967295 port=qmax offset=slave

	#pragma HLS INTERFACE s_axilite port=relu_en
	#pragma HLS INTERFACE s_axilite port=CHout
	#pragma HLS INTERFACE s_axilite port=Sx
	#pragma HLS INTERFACE s_axilite port=Hin
	#pragma HLS INTERFACE s_axilite port=CHin
	#pragma HLS INTERFACE s_axilite port=Kx
	#pragma HLS INTERFACE s_axilite port=mode
	#pragma HLS INTERFACE s_axilite port=Sy
	#pragma HLS INTERFACE s_axilite port=Ky
	#pragma HLS INTERFACE s_axilite port=Win
	#pragma HLS INTERFACE s_axilite port=return

//#pragma HLS ARRAY_PARTITION variable=feature_in cyclic factor=3 dim=1 partition
//#pragma HLS ARRAY_PARTITION variable=W cyclic factor=3 dim=1 partition
	ap_uint<8> pad_x,pad_y;
	if (mode == 0)
	{
		pad_x = 0;
		pad_y = 0;
	}
	else
	{
		pad_x = (Kx - 1) / 2;
		pad_y = (Ky - 1) / 2;
	}
	ap_uint<16> Wout = (Win + 2*pad_x - Kx) / Sx + 1;
	ap_uint<16> Hout = (Hin + 2*pad_y - Ky) / Sy + 1;
//#pragma HLS unroll
	channel:
	for (int cout = 0; cout < CHout; cout++)
	{
		ffeature_row:
		for (int i = 0; i < Hout; i++)
		{
			feature_col:
			for (int j = 0; j < Wout; j++)
			{
//#pragma HLS pipeline II=1
				ap_int<32> sum = 0;
				weight_row:
				for (int ii = 0; ii < Ky; ii++)
				{
//#pragma HLS unroll factor=2
					weight_col:
					for (int jj = 0; jj < Kx; jj++)
					{
//#pragma HLS PIPELINE
//#pragma HLS unroll factor=3
						ap_uint<16> h = i*Sy - pad_y + ii;
						ap_uint<16> w = j*Sx - pad_x + jj;
						if (h>=0 && w>=0 && h<Hin && w<Win)
						{
							Input_Channel:
							for (int cin = 0; cin < CHin; cin++)
							{
								//Feature [H][W][C]
								//kernel: [Ky][Kx][CHin][CHout]
								//Dtype_mul tp=feature_in[h][w][cin]*w[ii][jj][cin][cout];
								//std::cout<<"h:"<<h<<",w"<<w<<",cin"<<cin<<"\n";
								//std::cout<<"feature_in["<<h*CHin*Win+w*CHin+cin<<"]*W["<<ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout<<"]\n";
#pragma HLS pipeline II=2
//#pragma HLS unroll factor=16
								sum+=feature_in[h*CHin*Win+w*CHin+cin]*W[ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout];
							}
						}
					}
				}
				Dtype_f tmp = (Dtype_f)(sum*qmax[0]*qmax[1]/(127*127))+bias[cout];
				if(relu_en && tmp < 0)
					tmp = 0;
				feature_out[i*Wout*CHout+j*CHout+cout] = tmp;
			}
		}
	}
}
