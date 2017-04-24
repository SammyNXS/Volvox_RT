/*
/* Volvox algae rendering project. Modified from example work by NVIDIA. Original
   copyright retained below.
*/

/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#  include <GL/wglew.h>
#  include <GL/freeglut.h>
#  else
#  include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "commonStructs.h"
#include "random.h"
#include <Arcball.h>

#include <sstream>
#include <direct.h>

#include "modelLoader.h"

#define USING_THIN_LENS
#define USING_TRANSPARENCY

#define VOLVOX_LEVEL 3
#define D_VOLVOX_LEVEL 3

#define FOCAL_LENGTH 4.f
#define FOCAL_LENGTH_INCR 0.2f

#define MIN_FOCAL_LENGTH 0.f
#define MAX_FOCAL_LENGTH 100.f

#define LENS_RAD 0.f
#define LENS_RAD_INCR 0.2f

#define RES_WIDTH 810u//540u   1080u

#define RES_HEIGHT 540u//360u    720u


using namespace optix;
using namespace std;

const char* const SAMPLE_NAME = "Volvox";


static float rand_range(float min, float max)
{
    static unsigned int seed = 0u;
    return min + (max - min) * rnd(seed);
}

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context      context;
uint32_t     width  = RES_WIDTH;
uint32_t     height = RES_HEIGHT;
bool         use_pbo = true;

std::string  texture_path;
std::string  render_ptx_path;

// Camera state
float3       camera_up;
float3       camera_lookat;
float3       camera_eye;
Matrix4x4    camera_rotate;
sutil::Arcball arcball;

// Mouse state
int2       mouse_prev_pos;
int        mouse_button;

// Materials
Material volvox_matl;
Material d_volvox_matl;
Material diffuse_matl;

float focal_length;

float lens_rad;

bool test_scale;

//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

std::string ptxPath( const std::string& cuda_file );
Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void setupMaterials();
void createGeometry();
void createTopGroups(Context context,
	const Geometry& geometry,
	const Geometry& d_geometry,
	const vector<float3>* locs,
	const vector<float3>* d_locs,
	bool no_accel);
void setupCamera();
void setupLights();
void updateCamera();
void glutInitialize( int* argc, char** argv );
void glutRun();

void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutMousePress( int button, int state, int x, int y );
void glutMouseMotion( int x, int y);
void glutResize( int w, int h );


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

std::string ptxPath( const std::string& cuda_file )
{
	char buffer[BUFSIZ];

	// Get the current working directory:   
	char *answer = _getcwd(buffer, sizeof(buffer));
	std::string s_cwd;
	if (answer)
	{
		s_cwd = answer;
	}

    return std::string( s_cwd + "\\PTX_files\\" + cuda_file +".ptx");
}


Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}


void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
    atexit( destroyContext );
#endif
}


void createContext()
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 4640 );

    // Note: high max depth for reflection and refraction through Volvox
    context["max_depth"]->setInt( 100 );
    context["radiance_ray_type"]->setUint( 0 );
    context["shadow_ray_type"]->setUint( 1 );
    context["scene_epsilon"]->setFloat( 1.e-4f );
    context["importance_cutoff"]->setFloat( 0.01f );
    context["ambient_light_color"]->setFloat( 0.31f, 0.33f, 0.28f );

    // Output buffer
    // First allocate the memory for the GL buffer, then attach it to OptiX.
    GLuint vbo = 0;
    glGenBuffers( 1, &vbo );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glBufferData( GL_ARRAY_BUFFER, 4 * width * height, 0, GL_STREAM_DRAW);
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    Buffer buffer = 
		sutil::createOutputBuffer( context,
			RT_FORMAT_UNSIGNED_BYTE4,
			width,
			height,
			use_pbo );
    context["output_buffer"]->set( buffer );

    // Ray generation program
    {
#ifdef USING_THIN_LENS
		const std::string camera_name = "thin_lens_camera";
		focal_length = FOCAL_LENGTH;
		lens_rad = LENS_RAD;

		context["f_length"]->setFloat(focal_length);
		context["lens_rad"]->setFloat(lens_rad);

#else
		const std::string camera_name = "pinhole_camera";
#endif

        Program ray_gen_program = 
			context->createProgramFromPTXFile( render_ptx_path, camera_name );
        context->setRayGenerationProgram( 0, ray_gen_program );
    }

    // Exception program
    Program exception_program = 
		context->createProgramFromPTXFile( render_ptx_path, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );
	{
		const std::string miss_name = "miss";
		context->setMissProgram(0,
			context->createProgramFromPTXFile(render_ptx_path, miss_name));
		const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
		context["bg_color"]->setFloat(make_float3(0.f, 0.f, 0.f));
	}
}

void setupMaterials()
{
	// Materials
	diffuse_matl = context->createMaterial();
	Program diffuse_ch = 
		context->createProgramFromPTXFile(render_ptx_path,
			"closest_hit_radiance3");
	diffuse_matl->setClosestHitProgram(0, diffuse_ch);
	Program diff_ah = 
		context->createProgramFromPTXFile(render_ptx_path, "any_hit_shadow");
	diffuse_matl->setAnyHitProgram(1, diff_ah);
	diffuse_matl["Ka"]->setFloat(0.3f, 0.3f, 0.3f);
	diffuse_matl["Kd"]->setFloat(0.1608f, 0.7529f, 0.1333f);
	diffuse_matl["Ks"]->setFloat(0.8f, 0.9f, 0.8f);
	diffuse_matl["phong_exp"]->setFloat(88);
	diffuse_matl["reflectivity_n"]->setFloat(0.2f, 0.2f, 0.2f);

	// Volvox Material Color: 41,192,34 -> 0.160784, 0.75294117, 0.1333333
	Program volvox_ch = 
		context->createProgramFromPTXFile(render_ptx_path,
			"volvox_closest_hit_radiance");
	volvox_matl = context->createMaterial();
	volvox_matl->setClosestHitProgram(0, volvox_ch);

	volvox_matl["importance_cutoff"]->setFloat(1e-2f);
	volvox_matl["cutoff_color"]->setFloat(0.34f, 0.8f, 0.85f);

	volvox_matl["fresnel_exponent"]->setFloat(3.0f);
	volvox_matl["fresnel_minimum"]->setFloat(0.1f);
	volvox_matl["fresnel_maximum"]->setFloat(1.5f);

	volvox_matl["refraction_index"]->setFloat(1.0f);
	volvox_matl["refraction_color"]->setFloat(0.1608f, 0.7529f, 0.1333f);
	volvox_matl["refraction_maxdepth"]->setInt(100);

	volvox_matl["shadow_attenuation"]->setFloat(0.4f, 0.7f, 0.4f);
	volvox_matl["absorption"]->setFloat(0.2f);

	Program d_volvox_ch =
		context->createProgramFromPTXFile(render_ptx_path,
			"volvox_closest_hit_radiance");
	d_volvox_matl = context->createMaterial();
	d_volvox_matl->setClosestHitProgram(0, d_volvox_ch);

	d_volvox_matl["importance_cutoff"]->setFloat(1e-2f);
	d_volvox_matl["cutoff_color"]->setFloat(0.34f, 0.8f, 0.85f);

	d_volvox_matl["fresnel_exponent"]->setFloat(3.0f);
	d_volvox_matl["fresnel_minimum"]->setFloat(0.1f);
	d_volvox_matl["fresnel_maximum"]->setFloat(.5f);

	d_volvox_matl["refraction_index"]->setFloat(1.0f);
	d_volvox_matl["refraction_color"]->setFloat(0.1608f, 0.7529f, 0.1333f);
	d_volvox_matl["refraction_maxdepth"]->setInt(100);

	d_volvox_matl["shadow_attenuation"]->setFloat(0.4f, 0.7f, 0.4f);
	volvox_matl["absorption"]->setFloat(0.2f);

}

bool isEqual(float3 a, float3 b)
{
	return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

bool contains_float3(vector<float3>* vertices, float3 v)
{
	for (int i = 0; i < vertices->size(); i++)
	{
		const float3 vtemp = vertices->at(i);
		if (isEqual(vtemp,v))
		{
			return true;
		}
	}
	return false;
}

vector<float3>* loadIcosphere(string file)
{
	// Load and initialize a single teapot model
	string filepath = "obj\\" + file + ".obj";
	Model ico_model(filepath);
	ico_model.loadModel(filepath);

	vector<float3> verticesFull = ico_model.meshes[0].vertices;
	vector<float3>* vertices = new vector<float3>();

	for (int i = 0; i < verticesFull.size(); i++)
	{
		const float3 v = verticesFull[i];
		if(!contains_float3(vertices,v))
		{
			vertices->push_back(v);
		}
	}

	return vertices;
}

void createGeometry()
{
	// Load icosphere here

	string icosphere_file = "icosphere_" + to_string(VOLVOX_LEVEL);

	string d_icosphere_file = "icosphere_" + to_string(D_VOLVOX_LEVEL);


	vector<float3>* sphere_locs = loadIcosphere(icosphere_file);

	vector<float3>* d_sphere_locs = loadIcosphere(d_icosphere_file);

	const std::string sphere_ptx(ptxPath("sphere.cu"));
	Program sphere_bounds = 
		context->createProgramFromPTXFile(sphere_ptx, "bounds");
	Program sphere_intersect = 
		context->createProgramFromPTXFile(sphere_ptx, "robust_intersect");

	float rad = (VOLVOX_LEVEL >= 3) ? 0.035 : 0.05;

	Geometry sphere = context->createGeometry();
	sphere->setPrimitiveCount(1u);
	sphere->setBoundingBoxProgram(sphere_bounds);
	sphere->setIntersectionProgram(sphere_intersect);
	sphere["sphere"]->setFloat(make_float4(0., 0., 0., rad));

	rad = 0.08;

	Geometry d_sphere = context->createGeometry();
	d_sphere->setPrimitiveCount(1u);
	d_sphere->setBoundingBoxProgram(sphere_bounds);
	d_sphere->setIntersectionProgram(sphere_intersect);
	d_sphere["sphere"]->setFloat(make_float4(0., 0., 0., rad));

	setupMaterials();
    
	createTopGroups(context, sphere, d_sphere, sphere_locs, d_sphere_locs, false);
}

std::vector<float4>* GenerateVolvoxLocs()
{
	std::vector<float4>* volvox_locs = new std::vector<float4>();

	volvox_locs->push_back(make_float4(0.f, 0.f, 7.5f, 0.5f));
	volvox_locs->push_back(make_float4(-2.f, -2.f, 11.f, 0.5f));
	volvox_locs->push_back(make_float4(-2.f, -4.f, 17.f, 0.5f));
	volvox_locs->push_back(make_float4(-1.f, 2.f, 8.f, 0.5f));
	volvox_locs->push_back(make_float4(3.f, 1.8f, 7.f, 0.5f));

	volvox_locs->push_back(make_float4(4.f, -3.f, 13.f, 0.5f));
	volvox_locs->push_back(make_float4(0.f, -2.f, 8.f, 0.5f));
	volvox_locs->push_back(make_float4(-1.f, -1.f, 8.f, 0.5f));
	volvox_locs->push_back(make_float4(-2.f, 0.f, 11.f, 0.5f));

	volvox_locs->push_back(make_float4(-3.f, 2.f, 13.f, 0.5f));
	volvox_locs->push_back(make_float4(2.5f, 2.f, 15.f, 0.5f));
	volvox_locs->push_back(make_float4(2.f, -2.f, 15.f, 0.5f));
	volvox_locs->push_back(make_float4(1.f, 2.f, 8.f, 0.5f));
	volvox_locs->push_back(make_float4(1.f, -2.f, 10.f, 0.5f));

	volvox_locs->push_back(make_float4(3.f, 0.f, 10.f, 0.5f));
	volvox_locs->push_back(make_float4(-1.f, 1.f, 6.f, 0.5f));
	volvox_locs->push_back(make_float4(1.8f, 1.f, 6.f, 0.5f));

	volvox_locs->push_back(make_float4(2.f, -2.f, 8.f, 0.5f));
	volvox_locs->push_back(make_float4(-0.2f, -0.5f, 6.f, 0.5f));
	volvox_locs->push_back(make_float4(-0.2f, -1.f, 10.f, 0.5f));

	volvox_locs->push_back(make_float4(-3.f, -1.f, 12.f, 0.5f));
	volvox_locs->push_back(make_float4(-2.f, -2.f, 14.f, 0.5f));
	volvox_locs->push_back(make_float4(-5.f, -3.f, 14.f, 0.5f));
	volvox_locs->push_back(make_float4(-1.5f, 0.5f, 5.f, 0.5f));
	volvox_locs->push_back(make_float4(2.f, 0.f, 9.f, 0.5f));

	return volvox_locs;
}

std::vector<float4>* GenerateDaugheterVolvoxLocs()
{
	std::vector<float4>* volvox_locs = new std::vector<float4>();

	// Volvox daughter cells
	volvox_locs->push_back(make_float4(2.f, 0.1f, 9.f, 0.15f));

	volvox_locs->push_back(make_float4(-2.2f, -3.9f, 17.1f, 0.15f));

	volvox_locs->push_back(make_float4(-1.4f, 0.4f, 4.9f, 0.16f));

	volvox_locs->push_back(make_float4(1.1f, -1.8f, 10.2f, 0.2f));

	volvox_locs->push_back(make_float4(-2.f, -2.f, 10.8f, 0.12f));

	volvox_locs->push_back(make_float4(0.f, -0.7f, 6.2f, 0.15f));

	return volvox_locs;
}

void createTopGroups(Context context,
	const Geometry& geometry,
	const Geometry& d_geometry,
	const vector<float3>* locs,
	const vector<float3>* d_locs,
	bool no_accel)
{
	// Geometry group acceleration
	Acceleration gg_accel = no_accel ?
		context->createAcceleration("NoAccel") :
		context->createAcceleration("Bvh");

	test_scale = false;

	// Sphere instance.
	GeometryInstance sphere_inst = context->createGeometryInstance();
	sphere_inst->setGeometry(geometry);
	sphere_inst->setMaterialCount(1);
#ifdef USING_TRANSPARENCY
	sphere_inst->setMaterial(0, volvox_matl);
#else
	sphere_inst->setMaterial(0, diffuse_matl);
#endif

	// Sphere instance.
	GeometryInstance d_sphere_inst = context->createGeometryInstance();
	d_sphere_inst->setGeometry(d_geometry);
	d_sphere_inst->setMaterialCount(1);
#ifdef USING_TRANSPARENCY
	d_sphere_inst->setMaterial(0, d_volvox_matl);
#else
	d_sphere_inst->setMaterial(0, diffuse_matl);
#endif

	// Wrap sphere instance in a geometry group, to be put into a 
	GeometryGroup geometry_group = context->createGeometryGroup();
	geometry_group->setChildCount(1);
	geometry_group->setChild(0, sphere_inst);
	geometry_group->setAcceleration(context->createAcceleration("Bvh"));

	Acceleration volvox_accel = no_accel ?
		context->createAcceleration("NoAccel") :
		context->createAcceleration("Trbvh");

	Group volvox_group = context->createGroup();
	volvox_group->setChildCount(static_cast<unsigned int>(locs->size()));

	for (int i = 0; i < locs->size(); ++i)
	{
		Transform transform = context->createTransform();
		transform->setChild(geometry_group);
		volvox_group->setChild(i, transform);
		float3 pos = locs->at(i);
		float m[16] = { 1,0,0,-pos.x,
			0,1,0,-pos.y,
			0,0,1,-pos.z,
			0,0,0,1 };
		transform->setMatrix(false, NULL, m);
	}
	volvox_group->setAcceleration(volvox_accel);

	// Wrap sphere instance in a geometry group, to be put into a 
	GeometryGroup d_geometry_group = context->createGeometryGroup();
	d_geometry_group->setChildCount(1);
	d_geometry_group->setChild(0, d_sphere_inst);
	d_geometry_group->setAcceleration(context->createAcceleration("Bvh"));

	Acceleration d_volvox_accel = no_accel ?
		context->createAcceleration("NoAccel") :
		context->createAcceleration("Trbvh");

	Group d_volvox_group = context->createGroup();
	d_volvox_group->setChildCount(static_cast<unsigned int>(d_locs->size()));

	for (int i = 0; i < d_locs->size(); ++i)
	{
		Transform transform = context->createTransform();
		transform->setChild(d_geometry_group);
		d_volvox_group->setChild(i, transform);
		float3 pos = d_locs->at(i);
		float m[16] = { 1,0,0,-pos.x,
			0,1,0,-pos.y,
			0,0,1,-pos.z,
			0,0,0,1 };
		transform->setMatrix(false, NULL, m);
	}
	d_volvox_group->setAcceleration(d_volvox_accel);

	std::vector<float4>* volvox_locs = GenerateVolvoxLocs();

	std::vector<float4>* d_volvox_locs = GenerateDaugheterVolvoxLocs();

	// Create a toplevel group holding all the row groups.
	Acceleration top_accel = no_accel ?
		context->createAcceleration("NoAccel") :
		context->createAcceleration("Bvh");
	Group top_group = context->createGroup();;
	top_group->setChildCount(static_cast<unsigned int>(volvox_locs->size() + d_volvox_locs->size()));// +static_cast<unsigned int>(d_volvox_locs->size()));
	for (unsigned int i = 0; i < volvox_locs->size(); i++) {
		Transform transform = context->createTransform();

		transform->setChild(volvox_group);

		float4 pos = volvox_locs->at(i);
		float m[16] = { 1/pos.w ,0,0,-pos.x / pos.w,
			0,1/pos.w ,0,-pos.y/ pos.w,
			0,0,1/pos.w ,-pos.z/ pos.w,
			0,0,0, 1};
		transform->setMatrix(false, NULL, m);
		top_group->setChild(i, transform);
	}

	for (unsigned int i = 0; i < d_volvox_locs->size(); i++) {
		Transform transform = context->createTransform();

		transform->setChild(d_volvox_group);

		float4 pos = d_volvox_locs->at(i);
		float m[16] = { 1 / pos.w ,0,0,-pos.x / pos.w,
			0,1 / pos.w ,0,-pos.y / pos.w,
			0,0,1 / pos.w ,-pos.z / pos.w,
			0,0,0, 1 };
		transform->setMatrix(false, NULL, m);
		top_group->setChild(i+volvox_locs->size(), transform);
	}


	top_group->setAcceleration(top_accel);

	// Attach to context
	context["top_object"]->set(top_group);
	context["top_shadower"]->set(top_group);
}



void setupCamera()
{
	camera_eye = make_float3(0.0f, 0.0f, 0.0f);
	camera_lookat = make_float3(0.0f, 0.0f, 1.0f);
	
	camera_up     = make_float3( 0.0f, 1.0f,  0.0f );

    camera_rotate  = Matrix4x4::identity();
}


void setupLights()
{
	BasicLight lights[] = {
		{ make_float3(0.f, -5.0f, -5.0f), make_float3(2.0f, 2.0f, 2.0f), 1 }
	};
    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( BasicLight ) );
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    context[ "lights" ]->set( light_buffer );


	volvox_matl["lights"]->set(light_buffer);
}


void updateCamera()
{
	const float vfov = 40.0f;
    const float aspect_ratio = static_cast<float>(width) /
                               static_cast<float>(height);

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    const Matrix4x4 frame = Matrix4x4::fromBasis(
            normalize( camera_u ),
            normalize( camera_v ),
            normalize( -camera_w ),
            camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans   = frame*camera_rotate*camera_rotate*frame_inv;

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    context["eye"]->setFloat( camera_eye );
    context["U"  ]->setFloat( camera_u );
    context["V"  ]->setFloat( camera_v );
    context["W"  ]->setFloat( camera_w );
}


void glutInitialize( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( SAMPLE_NAME );
    glutHideWindow();
}


void glutRun()
{
    // Initialize GL state
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1 );

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, width, height);

    glutShowWindow();
    glutReshapeWindow( width, height);

    // register glut callbacks
    glutDisplayFunc( glutDisplay );
    glutIdleFunc( glutDisplay );
    glutReshapeFunc( glutResize );
    glutKeyboardFunc( glutKeyboardPress );
    glutMouseFunc( glutMousePress );
    glutMotionFunc( glutMouseMotion );

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
    updateCamera();

    context->launch( 0, width, height );

    Buffer buffer = getOutputBuffer();
    sutil::displayBufferGL( getOutputBuffer() );

    {
        static unsigned frame_count = 0;
        sutil::displayFps( frame_count++ );
    }

    glutSwapBuffers();
}


void glutKeyboardPress( unsigned char k, int x, int y )
{

    switch( k )
    {
        case( 'q' ):
        case( 27 ): // ESC
        {
            destroyContext();
            exit(0);
        }
        case( 'p' ):
        {
            const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
            break;
        }
		case( 'w'): // Increase focal length
		{
			focal_length += FOCAL_LENGTH_INCR;

			if (focal_length > MAX_FOCAL_LENGTH)
				focal_length = MAX_FOCAL_LENGTH;

			context["f_length"]->setFloat(focal_length);
			break;
		}
		case('s'): // decrease focal length
		{
			focal_length -= FOCAL_LENGTH_INCR;
			
			if (focal_length < MIN_FOCAL_LENGTH)
				focal_length = MIN_FOCAL_LENGTH;

			context["f_length"]->setFloat(focal_length);
			break;
		}
		case('d'): // Increase lens radius
		{
			lens_rad += LENS_RAD_INCR;
			context["lens_rad"]->setFloat(lens_rad);
			break;
		}
		case('a'): // decrease lens radius
		{
			lens_rad -= LENS_RAD_INCR;
			if (lens_rad < 0.f) 
				lens_rad = 0.f;
			context["lens_rad"]->setFloat(lens_rad);
			break;
		}
    }
}


void glutMousePress( int button, int state, int x, int y )
{
    if( state == GLUT_DOWN )
    {
        mouse_button = button;
        mouse_prev_pos = make_int2( x, y );
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion( int x, int y)
{
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
        const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
                         static_cast<float>( width );
        const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
                         static_cast<float>( height );
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float scale = fminf( dmax, 0.9f );
        camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to   = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x   / width, to.y   / height };

        camera_rotate = arcball.rotate( b, a );
    }

    mouse_prev_pos = make_int2( x, y );
}


void glutResize( int w, int h )
{
    if ( w == (int)width && h == (int)height ) return;

    width  = w;
    height = h;

    sutil::resizeBuffer( getOutputBuffer(), width, height );

    glViewport(0, 0, width, height);

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char** argv )
{
    std::string out_file;

    std::stringstream ss;
	ss << "RenderFuncs.cu";
    render_ptx_path = ptxPath( ss.str() );

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        createContext();
        createGeometry();
        setupCamera();
        setupLights();

        context->validate();

        if ( out_file.empty() )
        {
            glutRun();
        }
        else
        {
            updateCamera();
            context->launch( 0, width, height );
            sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
            destroyContext();
        }
        return 0;
    }
    SUTIL_CATCH( context->get() )
}

