/*
	ASSIMP model loader for OBJ files.
*/


#include"modelLoader.h"

Mesh::Mesh(vector<float3> vertices,vector<int> indices,int noOfTriangles)
{
	this->vertices = vertices; 
	this->indices = indices;
	this->noOfTriangles = noOfTriangles;
}


void Model::loadModel(string path)
{
	Assimp::Importer import; 
	const aiScene * scene = import.ReadFile(path,0);
	if( !scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE||!scene->mRootNode)
	{
		cout<<"ERRPR::ASSIMP:: "<<import.GetErrorString()<<endl;
		return;
	}

	cout<<scene->mRootNode<<"\n";
	processNode(scene->mRootNode,scene);	
		
}

void Model::processNode(aiNode* node,const aiScene *scene)
{
	cout<<"meshNum: "<<scene->mNumMeshes<<endl;

	//In our case there are not mesh object we just have a single one so just processing that
	for( int i = 0 ; i < scene->mNumMeshes; i++)
	{
		aiMesh  *mesh = scene->mMeshes[0];

		cout<<mesh<<" no of Vertices "<<mesh->mNumVertices<<endl;
		meshes.push_back(processMesh(mesh,scene));
	}
}


Mesh Model::processMesh(aiMesh *mesh,const aiScene *scene)
{
	cout<<"Processed mesh "<<endl;
	vector<float3> vertices;
	vector<int> indices; 

	for(int i = 0 ; i<mesh->mNumVertices;i++)
	{
		float3 vertex;
		vertex.x = mesh->mVertices[i].x;
		vertex.y = mesh->mVertices[i].y;
		vertex.z = mesh->mVertices[i].z;

		vertices.push_back(vertex);	
	}

	cout<<"Faces "<<mesh->mNumFaces<<endl;
	for( int i = 0; i<mesh->mNumFaces;i++)
	{
		aiFace face = mesh->mFaces[i];
		for(int j = 0;j<face.mNumIndices;j++)
		{
			//cout<<i<<" "<<j<<" "<<face.mIndices[j]<<endl;
			indices.push_back(face.mIndices[j]);
		}
	}

	//final triangles have been read into understandable form
	return Mesh(vertices,indices,mesh->mNumFaces);
}


void Mesh::makeTriangle()
{
	triangleStore = (Triangle *)malloc(sizeof(Triangle)*noOfTriangles);

	int vCount = 0;
	cout<<"No Of Triangles "<<noOfTriangles<<endl;
	
	aabb.minmaxX.x = 1e11;
	aabb.minmaxX.y = -1e11;
	aabb.minmaxY.x = 1e11;
	aabb.minmaxY.y = -1e11;
	aabb.minmaxZ.x = 1e11;
	aabb.minmaxZ.y = -1e11;
	aabb.numTriangle = noOfTriangles; 	

	for( int i = 0;i<noOfTriangles;i++)
	{
		for( int j = 0 ; j < 3; j++)
		{
			triangleStore[i].vertices[j].x = vertices[vCount].x;
			triangleStore[i].vertices[j].y = vertices[vCount].y;
			triangleStore[i].vertices[j].z = vertices[vCount].z;


			if( vertices[vCount].x < aabb.minmaxX.x)
			{
				aabb.minmaxX.x = vertices[vCount].x;
			}
			else if(vertices[vCount].x > aabb.minmaxX.y )
			{
				aabb.minmaxX.y = vertices[vCount].x;
			}
			else
			{}

			if( vertices[vCount].y < aabb.minmaxY.x)
			{
				aabb.minmaxY.x = vertices[vCount].y;
			}
			else if(vertices[vCount].y > aabb.minmaxY.y )
			{
				aabb.minmaxY.y = vertices[vCount].y;
			}
			else
			{}

			if( vertices[vCount].z < aabb.minmaxZ.x)
			{
				aabb.minmaxZ.x = vertices[vCount].z;
			}
			else if(vertices[vCount].z > aabb.minmaxZ.y )
			{
				aabb.minmaxZ.y = vertices[vCount].z;
			}
			else
			{}
			vCount++;
		}	
	}
	


	

}

