


#include "gvdb.h"
#include "file_png.h"
#include "file_tga.h"
#include "hdr_loader.h"

#include <stdlib.h>
#include <stdio.h>

//Cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Sample utils
#include "main.h"			// window system 
#include <GL/glew.h>

VolumeGVDB	gvdb;
CUmodule		cuCustom;
CUfunction		cuRaycastKernel;


bool cudaCheck(CUresult status, char* msg)
{
	if (status != CUDA_SUCCESS) {
		const char* stat = "";
		cuGetErrorString(status, &stat);
		printf("CUDA ERROR: %s (in %s)\n", stat, msg);
		exit(-1);
		return false;
	}
	return true;
}

bool cudaCheck(cudaError_t error, char* msg)
{
	if (!error) {
		const char* stat = "";
		printf("CUDA ERROR: %s (in %s)\n", stat, msg);
		exit(-1);
		return false;
	}
	return true;
}

class Sample : public NVPWindow {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);
	virtual void motion(int x, int y, int dx, int dy);
	virtual void mouse(NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y);
	//virtual bool createEnvironment(cudaTextureObject_t *env_tex, cudaArray_t *env_tex_data, const char *env_map_name);
	int			gl_screen_tex;
	int			mouse_down;

	Vector3DF	m_pretrans, m_scale, m_angs, m_trans;
};


bool Sample::init() {

	int w = getWidth(), h = getHeight();			// window width & height
	mouse_down = -1;
	gl_screen_tex = -1;
	m_pretrans.Set(-125, -160, -125);
	m_scale.Set(1, 1, 1);
	m_angs.Set(0, 0, 0);
	m_trans.Set(0, 0, 0);

	/*
	//Create environment texture
	cudaTextureObject_t env_tex;
	memset(&env_tex, 0, sizeof(cudaTextureObject_t));
	cudaArray_t env_tex_data = 0;
	createEnvironment(&env_tex, &env_tex_data, "D:/HDRI/Barce_Rooftop_C_3k.hdr");
	*/

	// Initialize GVDB	
	gvdb.SetVerbose(true);
	gvdb.SetCudaDevice(GVDB_DEV_FIRST);
	gvdb.Initialize();
	gvdb.AddPath("../source/shared_assets/");
	gvdb.AddPath("../shared_assets/");
	gvdb.AddPath(ASSET_PATH);

	// Load VDB file

	char scnpath[1024];

	if (!gvdb.FindFile("wdas_cloud_eighth.vdb", scnpath)) {
		printf("Cannot find vdb file.\n");
		exit(-1);
	}
	printf("Loading VDB. %s\n", scnpath);
	gvdb.SetChannelDefault(16, 16, 16);
	gvdb.LoadVDB(scnpath);
	gvdb.Measure(true);
	gvdb.getScene()->SetVolumeRange(0.1f, 0.0f, 1.0f);
	gvdb.getScene()->SetBackgroundClr(0.1f, 0.2f, 0.4f, 1.0);

	// Create Camera and Light
	Camera3D* cam = new Camera3D;
	cam->setFov(100);
	cam->setOrbit(Vector3DF(20, 30, 0), Vector3DF(125, 160, 125), 600, 1.0f);
	gvdb.getScene()->SetCamera(cam);
	gvdb.getScene()->SetRes(w, h);

	Light* lgt = new Light;
	lgt->setOrbit(Vector3DF(50, 65, 0), Vector3DF(125, 140, 125), 200, 1.0f);
	gvdb.getScene()->SetLight(0, lgt);


	// Add render buffer 
	printf("Creating screen buffer. %d x %d\n", w, h);
	gvdb.AddRenderBuf(0, w, h, 4);


	// Load custom module and kernel
	printf("Loading module: render_custom.ptx\n");
	cudaCheck(cuModuleLoad(&cuCustom, "pathTrace.ptx"), "cuModuleLoad (render_custom)");
	cudaCheck(cuModuleGetFunction(&cuRaycastKernel, cuCustom, "pathTrace"), "cuModuleGetFunction (pathTrace)");

	// Set GVDB to custom module 
	gvdb.SetModule(cuCustom);


	// Create opengl texture for display
	// This is a helper func in sample utils (not part of gvdb),
	// which creates or resizes an opengl 2D texture.
	createScreenQuadGL(&gl_screen_tex, w, h);

	return true;

}


void Sample::reshape(int w, int h)
{
	// Resize the opengl screen texture
	glViewport(0, 0, w, h);
	createScreenQuadGL(&gl_screen_tex, w, h);

	// Resize the GVDB render buffers
	gvdb.ResizeRenderBuf(0, w, h, 4);

	postRedisplay();
}

void Sample::display()
{
	gvdb.RenderKernel(cuRaycastKernel, 0, 0);
	//gvdb.Render(SHADE_VOLUME, 0, 0);
	// Copy render buffer into opengl texture
	// This function does a gpu-gpu device copy from the gvdb cuda output buffer
	// into the opengl texture, avoiding the cpu readback found in ReadRenderBuf
	gvdb.ReadRenderTexGL(0, gl_screen_tex);

	// Render screen-space quad with texture
	// This is a helper func in sample utils (not part of gvdb),
	// which renders an opengl 2D texture to the screen.
	renderScreenQuadGL(gl_screen_tex);

	postRedisplay();
}

void Sample::motion(int x, int y, int dx, int dy)
{
	// Get camera for GVDB Scene
	Camera3D* cam = gvdb.getScene()->getCamera();

	switch (mouse_down) {
	case NVPWindow::MOUSE_BUTTON_LEFT: {
		// Adjust orbit angles
		Vector3DF angs = cam->getAng();
		angs.x += dx * 0.2f;
		angs.y -= dy * 0.2f;
		cam->setOrbit(angs, cam->getToPos(), cam->getOrbitDist(), cam->getDolly());
		postRedisplay();	// Update display
	} break;

	case NVPWindow::MOUSE_BUTTON_MIDDLE: {
		// Adjust target pos		
		cam->moveRelative(float(dx) * cam->getOrbitDist() / 1000, float(-dy) * cam->getOrbitDist() / 1000, 0);
		postRedisplay();	// Update display
	} break;

	case NVPWindow::MOUSE_BUTTON_RIGHT: {
		// Adjust dist
		float dist = cam->getOrbitDist();
		dist -= dy;
		cam->setOrbit(cam->getAng(), cam->getToPos(), dist, cam->getDolly());
		postRedisplay();	// Update display
	} break;
	}
}

void Sample::mouse(NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y)
{
	// Track when we are in a mouse drag
	mouse_down = (state == NVPWindow::BUTTON_PRESS) ? button : -1;
}

/*
bool Sample::createEnvironment(cudaTextureObject_t * env_tex, cudaArray_t * env_tex_data, const char * env_map_name)
{
	//from ray tracing gems Ch_28

	unsigned int rx, ry;
	float *pixels; 

	if (!load_hdr_float4(&pixels, &rx, &ry, env_map_name)) {
		fprintf(stderr, "error loading environment mapfile %s\n", env_map_name);
		return false; 
	}

	const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
	cudaCheck(cudaMallocArray(env_tex_data, &channel_desc, rx, ry), "cudaMallocArray(env_tex_data, &channel_desc, rx, ry)");
	cudaCheck(cudaMemcpyToArray(*env_tex_data, 0, 0, pixels, rx*ry * sizeof(float4), cudaMemcpyHostToDevice), "cudaMemcpyToArray(*env_tex_data)");
	
	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = *env_tex_data;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaAddressModeWrap;
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.addressMode[2] = cudaAddressModeWrap;
	tex_desc.filterMode = cudaFilterModeLinear;
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = 1;

	cudaCheck(cudaCreateTextureObject(env_tex, &res_desc, &tex_desc, NULL), "cudaCreateTextureObject(env_tex, &res_desc, &tex_desc, NULL)");


	return true;
}

*/


int sample_main(int argc, const char** argv)
{
	Sample sample_obj;
	return sample_obj.run("gPathTrace", "pt", argc, argv, 1024, 768, 4, 5);
}

void sample_print(int argc, char const *argv)
{
}