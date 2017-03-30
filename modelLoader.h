#pragma once

#include<iostream>
#include<vector>
#include<stdio.h>
#include <assimp\include\assimp\Importer.hpp>
#include<assimp\include\assimp\scene.h>
#include<assimp\include\assimp\postprocess.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>


using namespace std; 
using namespace optix;


#define TEAPOTNUM 4 


struct Triangle{
	float3 vertices[3];
	float3 normal; //this will have to be calculated
};


struct testAabb{
	float2 minmaxX;
	float2 minmaxY;
	float2 minmaxZ;
	int numTriangle;
};

struct Texture{
	int id; 
	string type;
};

class Mesh{
	public:
	vector<float3> vertices; 
	vector<int> indices; 
	vector<Texture> textures;
	Triangle *triangleStore;
	int noOfTriangles;
	//bounding box for the triangle
	testAabb aabb;
       	Mesh(vector<float3> vertices,vector<int> indices,int);	
	void makeTriangle();
};

class Model
{
	public:
	string path; 
	Model(string pat):path(pat){};

	//private: 
	vector<Mesh> meshes;
	string directory;
	void loadModel(string path);
	void processNode(aiNode * node,const aiScene* scene);
 	Mesh processMesh(aiMesh *mesh,const aiScene *scene);
};