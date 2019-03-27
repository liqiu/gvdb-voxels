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
//       Contents are modified from "cuda_gvdb_raycast.cuh" in gvdb library
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



// gvdbBrickFunc ( gvdb, channel, nodeid, t, pos, dir, pstep, hit, norm, clr )
typedef void(*gvdbBrickFunc_t)(VDBInfo*, uchar, int, float3, float3, float3, float3&, float3&, float3&, float4&);

#define MAXLEV			5
#define MAX_ITER		256
#define EPS				0.0001

#define LO		0
#define	MID		1.0
#define	HI		2.0
#define M_PI       3.14159265358979323846   // pi


// Helper functions 
inline __device__ float getLinearDepth(float* depthBufFloat)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;					// Pixel coordinates
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float z = depthBufFloat[(SCN_HEIGHT - 1 - y) * SCN_WIDTH + x];	// Get depth value
	float n = scn.camnear;
	float f = scn.camfar;
	return (-n * f / (f - n)) / (z - (f / (f - n)));				// Return linear depth
}

inline __device__ uchar4 getColor(VDBInfo* gvdb, uchar chan, float3 p)
{
	return tex3D<uchar4>(gvdb->volIn[chan], (int)p.x, (int)p.y, (int)p.z);
}
inline __device__ float4 getColorF(VDBInfo* gvdb, uchar chan, float3 p)
{
	return make_float4(tex3D<uchar4>(gvdb->volIn[chan], (int)p.x, (int)p.y, (int)p.z));
}



// Brick sampler

__device__ void customRayDeepBrick(VDBInfo* gvdb, uchar chan, int nodeid, float3 t, float3 pos, float3 dir, float3& pstep, float3& hit, float3& norm, float4& clr)
{
	float3 vmin;
	VDBNode* node = getNode(gvdb, 0, nodeid, &vmin);			// Get the VDB leaf node		

	//t.x = SCN_PSTEP * ceil( t.x / SCN_PSTEP );						// start on sampling wavefront	

	float3 o = make_float3(node->mValue);					// atlas sub-volume to trace
	float3 wp = pos + t.x*dir;
	float3 p = (wp - vmin) / gvdb->vdel[0];					// sample point in index coords	
	float3 wpt = SCN_PSTEP * dir * gvdb->vdel[0];					// world increment
	float4 val = make_float4(0, 0, 0, 0);
	float4 hclr;
	int iter = 0;
	float dt = length(SCN_PSTEP*dir*gvdb->vdel[0]);

	// record front hit point at first significant voxel
	if (hit.x == 0) hit.x = t.x; // length(wp - pos);

	// skip empty voxels
	for (iter = 0; val.w < SCN_MINVAL && iter < MAX_ITER && p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0]; iter++) {
		val.w = transfer(gvdb, tex3D<float>(gvdb->volIn[chan], p.x + o.x, p.y + o.y, p.z + o.z)).w;
		p += SCN_PSTEP * dir;
		wp += wpt;
		t.x += dt;
	}

	for (; iter < MAX_ITER && p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0]; iter++) {

		if (clr.x > 1 || clr.y > 1 || clr.z > 1 || clr.w > 1) return;
		val = transfer(gvdb, tex3D<float>(gvdb->volIn[chan], p.x + o.x, p.y + o.y, p.z + o.z));
		clr += val; 
		p += SCN_PSTEP * dir;
		wp += wpt;
		t.x += dt;

	}


	hit.y = t.x;  // length(wp - pos);

}

__device__ void myRayCast(VDBInfo* gvdb, uchar chan, float3 pos, float3 dir, float3& hit, float3& norm, float4& clr, gvdbBrickFunc_t brickFunc)
{
	int		nodeid[MAXLEV];					// level variables
	float	tMax[MAXLEV];
	int		b;

	// GVDB - Iterative Hierarchical 3DDA on GPU
	float3 vmin;
	int lev = gvdb->top_lev;
	nodeid[lev] = 0;		// rootid ndx
	float3 t = rayBoxIntersect(pos, dir, gvdb->bmin, gvdb->bmax);	// intersect ray with bounding box	
	VDBNode* node = getNode(gvdb, lev, nodeid[lev], &vmin);			// get root VDB node	
	if (t.z == NOHIT) return; //TODO:implement texture lookup here

	// 3DDA variables		
	t.x += gvdb->epsilon;
	tMax[lev] = t.y - gvdb->epsilon;
	float3 pStep = make_float3(isign3(dir));
	float3 p, tDel, tSide, mask;
	int iter;

	PREPARE_DDA

		for (iter = 0; iter < MAX_ITER && lev > 0 && lev <= gvdb->top_lev && p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x <= gvdb->res[lev] && p.y <= gvdb->res[lev] && p.z <= gvdb->res[lev]; iter++) {

			NEXT_DDA

				// depth buffer test [optional]
				if (SCN_DBUF != 0x0) {
					if (t.x > getLinearDepth(SCN_DBUF)) {
						hit.z = 0;
						return;
					}
				}

			// node active test
			b = (((int(p.z) << gvdb->dim[lev]) + int(p.y)) << gvdb->dim[lev]) + int(p.x);	// bitmaskpos
			if (isBitOn(gvdb, node, b)) {							// check vdb bitmask for voxel occupancy						
				if (lev == 1) {									// enter brick function..
					nodeid[0] = getChild(gvdb, node, b);
					t.x += gvdb->epsilon;
					(*brickFunc) (gvdb, chan, nodeid[0], t, pos, dir, pStep, hit, norm, clr);
					if (clr.w <= 0) {
						clr.w = 0;
						return;
					}			// deep termination				
					if (hit.z != NOHIT) return;						// surface termination												

					STEP_DDA										// leaf node empty, step DDA
					//t.x = hit.y;				
					//PREPARE_DDA

				}
				else {
					lev--;											// step down tree
					nodeid[lev] = getChild(gvdb, node, b);				// get child 
					node = getNode(gvdb, lev, nodeid[lev], &vmin);	// child node
					t.x += gvdb->epsilon;										// make sure we start inside child
					tMax[lev] = t.y - gvdb->epsilon;							// t.x = entry point, t.y = exit point							
					PREPARE_DDA										// start dda at next level down
				}
			}
			else {
				STEP_DDA											// empty voxel, step DDA
			}
			while (t.x > tMax[lev] && lev <= gvdb->top_lev) {
				lev++;												// step up tree
				if (lev <= gvdb->top_lev) {
					node = getNode(gvdb, lev, nodeid[lev], &vmin);
					PREPARE_DDA										// restore dda at next level up
				}
			}
		}
}


extern "C" __global__ void pathTrace(VDBInfo* gvdb, uchar chan, uchar4* outBuf) {


	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= scn.width || y >= scn.height) return;
	
	float3 rdir = normalize(getViewRay((float(x) + 0.5) / scn.width, (float(y) + 0.5) / scn.height));
	float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
	float4 clr = make_float4(0.1f, 0.1f, 0.1f, 0.1f);
	float3 norm;
	
	float4 density = make_float4(0,0,0,0); 

	myRayCast(gvdb, chan, scn.campos, rdir, hit, norm, clr, customRayDeepBrick);
	
	
	outBuf[y*scn.width + x] = make_uchar4(clr.x*255 , clr.y*255, clr.z*255, 1);


}