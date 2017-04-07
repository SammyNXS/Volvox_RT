/*
/*Render program functions, modified from work by NVIDIA
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

#include <optix.h>
#include <optix_math.h>
// Used by all the tutorial cuda files
#include "commonStructs.h"

using namespace optix;

#define FLT_MAX         1e30;

static __device__ __inline__ float3 exp( const float3& x )
{
  return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

static __device__ __inline__ float step( float min, float value )
{
  return value<min?0:1;
}

static __device__ __inline__ float3 mix( float3 a, float3 b, float x )
{
  return a*(1-x) + b*x;
}

static __device__ __inline__ float3 schlick( float nDi, const float3& rgb )
{
  float r = fresnel_schlick(nDi, 5, rgb.x, 1);
  float g = fresnel_schlick(nDi, 5, rgb.y, 1);
  float b = fresnel_schlick(nDi, 5, rgb.z, 1);
  return make_float3(r, g, b);
}

static __device__ __inline__ uchar4 make_color(const float3& c)
{
    return make_uchar4( static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
                        static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
                        static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
                        255u);                                                 /* A */
}


static __device__ __inline__ float2 concentric_sample_disk(const float2& uOffset)
{
	// ConcentricSampleDisk
	// a) Map uniform random number to [-1,1]^2
	//float2 uOffset = 2.f * u - make_float2(1.f, 1.f);

	// b) Handle degeneracy at origin
	if (uOffset.x == 0 && uOffset.y == 0)
		return make_float2(0.f, 0.f);

	// c) Apply concentric mapping to point
	float theta;
	float r;
	if(abs(uOffset.x) > abs(uOffset.y))
	{
		r = uOffset.x;
		theta = M_PI_4f * (uOffset.y / uOffset.x);
	}
	else
	{
		r = uOffset.y;
		theta = M_PI_2f - M_PI_4f * (uOffset.x / uOffset.y);
	}


	return r * make_float2(cos(theta), sin(theta));
}

static __device__ __inline__ float2 get_disk_sample(const float2& u,
	const float maxwidth,
	const float maxheight,
	const uchar2& index)
{

	float w = (-(1 / 2) + ((index.x + 0.5) / maxwidth));
		
	float h = ((1 / 2) - ((index.y + 0.5) / maxheight));

	// a) Map uniform random number to [-1,1]^2
	float2 uOffset = 2.f * u - make_float2(1.f, 1.f);

	float2 image_point = make_float2(w, h);

	image_point += uOffset;

	return concentric_sample_disk(image_point);
}



struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int depth;
};

struct PerRayData_shadow
{
  float3 attenuation;
};

