//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation. 
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
//    in the documentation and/or  other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived 
//    from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
// BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Version 1.0: Sergen Eren, 26/3/2019
//----------------------------------------------------------------------------------
// 
// File: Custom path trace kernel: 
//       Performs a custom ray marching inside volume
//
//-----------------------------------------------

#include <stdio.h>
#include "cuda_math.cuh"
#include <cuda_runtime.h> 
#include <curand_kernel.h>


typedef unsigned char		uchar;
typedef unsigned int		uint;
typedef unsigned short		ushort;
typedef unsigned long		ulong;
typedef unsigned long long	uint64;

//-------------------------------- GVDB Data Structure
#define CUDA_PATHWAY
#include "cuda_gvdb_scene.cuh"		// GVDB Scene
#include "cuda_gvdb_nodes.cuh"		// GVDB Node structure
#include "cuda_gvdb_geom.cuh"		// GVDB Geom helpers
#include "cuda_gvdb_dda.cuh"		// GVDB DDA 

#define MAXLEV			5
#define MAX_ITER		256
#define EPS				0.0001

#define LO				0
#define	MID				1.0
#define	HI				2.0
#define M_PI			3.14159265358979323846f   // pie
#define INV_4PI			1 / 4 * M_PI

// Helper functions 

inline __device__ uchar4 getColor(VDBInfo* gvdb, uchar chan, float3 p)
{
	return tex3D<uchar4>(gvdb->volIn[chan], (int)p.x, (int)p.y, (int)p.z);
}
inline __device__ float4 getColorF(VDBInfo* gvdb, uchar chan, float3 p)
{
	return make_float4(tex3D<uchar4>(gvdb->volIn[chan], (int)p.x, (int)p.y, (int)p.z));
}
inline __device__ float3 exp3(float3 val)
{
	float3 tmp = make_float3(exp(val.x), exp(val.y), exp(val.z));
	return tmp;
}

#define EPSTEST(a,b,c)	(a>b-c && a<b+c)
#define VOXEL_EPS	0.0001

//Phase functions 

__device__ float isotropic() {

	return INV_4PI;

}

__device__ float henyey_greenstein(float cos_theta, float g) {

	float denominator = 1 + g * g - 2 * g * cos_theta;

	return INV_4PI * (1 - g * g) / (denominator * sqrtf(denominator));

}

__device__ float double_henyey_greenstein(float cos_theta, float f, float g1, float g2) {

	return (1 - f)*henyey_greenstein(cos_theta, g1) + f * henyey_greenstein(cos_theta, g2);

}

__device__ float schlick(float cos_theta, float k) { // simpler hg phase function Note: -1<k<1   

	float denominator = 1 + k * cos_theta;

	return INV_4PI * (1 - k * k) / (denominator*denominator);

}

__device__ float rayleigh(float cos_sq_theta, float lambda) // rayleigh scattering
{

	return 3 * (1 + cos_sq_theta) / 4 * lambda*lambda*lambda*lambda; // 

}

__device__ float cornette_shanks(float cos_theta, float cos_sq_theta, float g) {

	float first_part = (1 - g * g) / (2 + g * g);
	float second_part = (1 + cos_sq_theta) / pow((1 + g * g - cos_theta), 1.5f);

	return INV_4PI * 1.5f * first_part * second_part;

}
// End phase functions


// Shadow ray marcher
__device__ float3 getShadowTransmittance(float3 pos, float sampledDistance, float stepSizeShadow, float3 extinction) {

	float3 shadow = make_float3(1.0f);
	float3 Ldir = normalize(scn.light_pos - pos);

	for (float tshadow = 0.0f; tshadow < sampledDistance; tshadow += stepSizeShadow) {

		float3 shadowPos = pos + Ldir * tshadow;
		float densityShadow = 1.0f;
		shadow *= exp3(-densityShadow * extinction*stepSizeShadow);
	}

	return shadow;

}

__device__ bool in_brick(VDBInfo* gvdb,  float3 pos) {

	return pos.x >= 0 && pos.y >= 0 && pos.z >= 0 && pos.x < gvdb->res[0] && pos.y < gvdb->res[0] && pos.z < gvdb->res[0];
}
__device__ void RayCast(VDBInfo* gvdb, uchar chan, float3 pos, float3 dir, float3& hit, float4& clr) {

	float3 absorption = 10.0f * make_float3(0.75, 0.5, 0.0);
	float3 scattering = 25.0f * make_float3(0.25, 0.5, 1.0);
	float3 extinction = absorption + scattering;

	float density = 0.0f;

	float3 scatteredLuminance = make_float3(0.0, 0.0, 0.0);
	float3 transmittance = make_float3(1.0);
	float3 L = make_float3(50, 50, 50); // Light color
	float3 color = make_float3(1.0, 0.0, 0.0);
	float stepSize = 0.005f;


	float3 t = rayBoxIntersect(pos, dir, gvdb->bmin, gvdb->bmax);
	if (t.z == NOHIT) return;

	float3 wpos = pos + dir * t.x; //get world position at first intersection 
	wpos += dir * 0.001; // add epsilon

	for (float f = t.x; f < t.y; f += stepSize) {

		if (transmittance.x < 0.1f) break; // no need to trace further

		//brick node variables 
		float3 vmin; //root pos of brick node
		uint64 nodeid; // brick id 
		float3 offset; // brick offset
		float3 vdel; // i.e. voxel size 

		VDBNode* brick_node = getNodeAtPoint(gvdb, wpos + dir * stepSize , &offset, &vmin, &vdel, &nodeid);  // Check if there is a brick node ahead of us 

		if (brick_node != 0x0) { //We have found a brick node in ray direction.

			//Find the entrance and exit points in brick node   
			float diag_len = sqrtf(vdel.x * vdel.x);												//              b.y
			float3 b = rayBoxIntersect(wpos, dir, vmin, vmin + diag_len * gvdb->res[0]);			//          ____._____                                  
			wpos += dir * b.x;																		//		    |  /      |
			float3 brick_pos = (wpos - vmin) / vdel;												//          | / dir   |
			brick_pos += dir * 0.001;																//	    b.x |/        |
																									//          |_________|

			float3 atlas_pos = make_float3(brick_node->mValue);					// Atlas space position of brick node
			
			// ray march brick
			for (int iter = 0; iter < MAX_ITER && in_brick(gvdb, brick_pos); iter++) { 
			
				density += tex3D<float>(gvdb->volIn[chan], brick_pos.x + atlas_pos.x, brick_pos.y + atlas_pos.y, brick_pos.z + atlas_pos.z); //Sample density at voxel 
				
				brick_pos += dir * stepSize;
				wpos += dir * stepSize * vdel;
			}
			density *= vdel.x;
			

		}
		wpos += dir * stepSize;
		transmittance *= make_float3(exp(-density * stepSize));

		//TODO: get shadow transmittance and evaluate clr by albedo and extinction coefficients  
		/*  
		// calculate accumulated shadow transmittance
		float stepSizeShadow = 0.1;
		float3 shadow = getShadowTransmittance(wpos, 1.0, stepSizeShadow, extinction);

		float3 S = L * shadow * density * scattering;
		float3 sampleExtinction = make_float3(fmaxf(0.0000000001, (density * extinction).x), fmaxf(0.0000000001, (density * extinction).y), fmaxf(0.0000000001, (density * extinction).z));
		float3 Sint = (S - S * exp3(-sampleExtinction * stepSize)) / sampleExtinction;
		scatteredLuminance += transmittance * Sint;

		// Evaluate transmittance to view independentely
		transmittance *= exp3(-sampleExtinction * stepSize);
		*/

	}
	transmittance = make_float3(fminf(fmaxf(transmittance.x, 0.001), 1.0f));

	clr = make_float4(transmittance, (1- transmittance.x) * 0.1);

}



extern "C" __global__ void pathTrace(VDBInfo* gvdb, uchar chan, uchar4* outBuf) {


	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= scn.width || y >= scn.height) return;

	float3 rdir = normalize(getViewRay((float(x) + 0.5) / scn.width, (float(y) + 0.5) / scn.height));
	float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
	float4 clr = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

	RayCast(gvdb, chan, scn.campos, rdir, hit, clr);

	outBuf[y*scn.width + x] = make_uchar4(clr.x * 255, clr.y * 255, clr.z * 255 , 255);


}