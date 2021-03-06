/* 
 /*Render program functions for Volvox. Modified from work by NVIDIA. Original
 copyright retained below.
 */

/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#include "RenderFuncs.h"
#include <optixu/optixu_aabb.h>
#include "random.h"

#define SAMPLE_ITERS_AXIS 3.f

#define SAMPLE_ITERS_RECIP (1.f / (SAMPLE_ITERS_AXIS * SAMPLE_ITERS_AXIS))

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type , , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(rtObject,     top_object, , );

//
// Pinhole camera implementation
//
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtBuffer<uchar4, 2>              output_buffer;

RT_PROGRAM void pinhole_camera()
{
  size_t2 screen = output_buffer.size();

  float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);

  optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);

  output_buffer[launch_index] = make_color( prd.result );
}

/*
	Thin lens camera, to produce simple focal plane/depth of field effect. Based
	primarily on implementation shown in Physically Based Rendering by Pharr, 
	Jakob, and Humphreys
*/

rtDeclareVariable(float, f_length, , );
rtDeclareVariable(float, lens_rad, , );

RT_PROGRAM void thin_lens_camera()
{
	// Get ray direction of eye to image plane in the same way as before
	size_t2 screen = output_buffer.size();

	float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
	float3 init_ray_origin = eye;
	float3 init_ray_direction = normalize(d.x*U + d.y*V + W);

	// If lens_radius is 0, treat as pinhole
	if(lens_rad == 0)
	{
		optix::Ray ray(init_ray_origin,
			init_ray_direction,
			radiance_ray_type,
			scene_epsilon);

		PerRayData_radiance prd;
		prd.importance = 1.f;
		prd.depth = 0;
		prd.delta_sum = 0.f;

		rtTrace(top_object, ray, prd);

		output_buffer[launch_index] = make_color(prd.result);
	}
	else
	{
		// Get distance
		float ft = f_length / dot(init_ray_direction,W);
		float3 pFocus = init_ray_origin + init_ray_direction * ft;

		float3 result_color = make_float3(0.f, 0.f, 0.f);

		// Sample across two dimensions of an NxN grid of jittered samples
		for (int i = 0; i < SAMPLE_ITERS_AXIS; i++)
		{
			for (int j = 0; j < SAMPLE_ITERS_AXIS; j++)
			{
				// 1) Sample point on lens
				unsigned seed = tea<4>(d.x, d.y);
				float2 sample_point = make_float2(rnd(seed), rnd(seed));

				float2 disc_point = get_disk_sample(sample_point,
					SAMPLE_ITERS_AXIS,
					SAMPLE_ITERS_AXIS,
					make_uchar2(i,j));

				float2 p_lens = lens_rad * disc_point;

				// 2) Compute point on plane of focus

				float3 ray_origin = init_ray_origin + p_lens.x * U + p_lens.y * V;
				float3 ray_direction = normalize(pFocus - ray_origin);

				// 3) Update ray for effect on lens
				optix::Ray ray(ray_origin,
					ray_direction,
					radiance_ray_type,
					scene_epsilon);

				PerRayData_radiance prd;
				prd.importance = 1.f;
				prd.depth = 0;
				prd.delta_sum = 0.f;

				rtTrace(top_object, ray, prd);
				result_color += prd.result;
			}
		}
		result_color *= SAMPLE_ITERS_RECIP;

		output_buffer[launch_index] = make_color(result_color);
	}
}


//
// Returns solid color for miss rays
//

rtDeclareVariable(float3, bg_color, , );
RT_PROGRAM void miss()
{
	prd_radiance.result = bg_color;
}

//
// Terminates and fully attenuates ray after any hit
//
RT_PROGRAM void any_hit_shadow()
{
	// this material is opaque, so it fully attenuates all shadow rays
	prd_shadow.attenuation = make_float3(0);

	rtTerminateRay();
}

//
// Attenuates shadow rays for shadowing transparent objects
//

rtDeclareVariable(float3, shadow_attenuation, , );

RT_PROGRAM void glass_any_hit_shadow()
{
  float3 world_normal = 
	  normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float nDi = fabs(dot(world_normal, ray.direction));

  prd_shadow.attenuation *=
	  1-fresnel_schlick(nDi, 5, 1-shadow_attenuation, make_float3(1));

  rtIgnoreIntersection();
}


/*
	Volvox transparency closest hit program. Based on NVidia OptiX example
	glass material program.
*/

rtDeclareVariable(float3,       cutoff_color, , );
rtDeclareVariable(float,        fresnel_exponent, , );
rtDeclareVariable(float,        fresnel_minimum, , );
rtDeclareVariable(float,        fresnel_maximum, , );
rtDeclareVariable(float,        refraction_index, , );
rtDeclareVariable(int,          refraction_maxdepth, , );
rtDeclareVariable(int,          reflection_maxdepth, , );
rtDeclareVariable(float3,       refraction_color, , );
rtDeclareVariable(float3,       reflection_color, , );
rtDeclareVariable(float, importance_cutoff, , );
rtDeclareVariable(float, absorption, , );

rtDeclareVariable(int, max_depth, , );
rtBuffer<BasicLight>        lights;

RT_PROGRAM void volvox_closest_hit_radiance()
{
  // intersection vectors
  const float3 h = ray.origin + t_hit * ray.direction;            // hitpoint
  const float3 n = 
	  normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal
  const float3 i = ray.direction;                           // incident direction

  float reflection = 0.5f;
  float3 result = make_float3(0.0f,0.1f, 0.f);

  // refraction
  if (prd_radiance.depth < min(refraction_maxdepth, max_depth))
  {
    float3 t;											// transmission direction
    // check for external or internal reflection
    float cos_theta = dot(i, n);

	bool isExiting = (cos_theta < 0.0f);

    if (isExiting)
		cos_theta = -cos_theta;

    reflection = fresnel_schlick(cos_theta,
		fresnel_exponent,
		fresnel_minimum,
		fresnel_maximum);

    float importance = 
		prd_radiance.importance 
		* (1.0f-reflection) 
		* optix::luminance( refraction_color );
	if ( importance > importance_cutoff ) 
	{
		/*
			Single Scattering Approximation
		*/
		// Construct ray as normal
		optix::Ray ray( h, t, radiance_ray_type, scene_epsilon );
		PerRayData_radiance refr_prd;
		refr_prd.depth = prd_radiance.depth+1;
		refr_prd.importance = importance;

		float delta = prd_radiance.delta_sum;
		
		float3 attenuation = make_float3(1.f);

		// Before sending ray, we must add delta distance to accumulation.
		// Only perform if the initial hit was entering (i.e., the new ray trace
		// will be exiting)
		if (!isExiting)
		{
			// Get delta distance between initial hit distance and hit distance.
			// Add to current accumulation of delta sum .
			delta += length(t - t_hit);
			refr_prd.delta_sum =  delta;
		}

		// Perform ray trace. This should allow for accumulation down of light and
		// delta summations
		rtTrace(top_object, ray, refr_prd);

		// This environment assumes only one light source
		BasicLight light = lights[0];

		// Get attenuated fraction of light for this hit location:
		// I_0 * e^-(a*delta)
		attenuation = light.color * exp2f(-absorption * delta);


		// Apply  attenuation on the final result color. The result from ray 
		// should already include accumulation from the farthest hit Volvox sphere
		result += (1.0f - reflection) * refraction_color * refr_prd.result * attenuation;
    }
    // else TIR
  }

  prd_radiance.result = result;
}


//
// (UPDATED)
// Phong surface shading with shadows 
//

rtDeclareVariable(float3, Ka, , );
rtDeclareVariable(float3, Ks, , );
rtDeclareVariable(float, phong_exp, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float3, ambient_light_color, , );
rtDeclareVariable(rtObject, top_shadower, , );

RT_PROGRAM void closest_hit_radiance3()
{
	float3 world_geo_normal = 
		normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = 
		normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = 
		faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 color = Ka * ambient_light_color;

	float3 hit_point = ray.origin + t_hit * ray.direction;

	for (int i = 0; i < lights.size(); ++i) {
		BasicLight light = lights[i];
		float3 L = normalize(light.pos - hit_point);
		float nDl = dot(ffnormal, L);

		if (nDl > 0.0f) {
			// cast shadow ray
			PerRayData_shadow shadow_prd;
			shadow_prd.attenuation = make_float3(1.0f);
			float Ldist = length(light.pos - hit_point);
			optix::Ray shadow_ray(hit_point,
				L,
				shadow_ray_type,
				scene_epsilon,
				Ldist);
			rtTrace(top_shadower, shadow_ray, shadow_prd);
			float3 light_attenuation = shadow_prd.attenuation;

			if (fmaxf(light_attenuation) > 0.0f) {
				float3 Lc = light.color * light_attenuation;
				color += Kd * nDl * Lc;

				float3 H = normalize(L - ray.direction);
				float nDh = dot(ffnormal, H);
				if (nDh > 0)
					color += Ks * Lc * pow(nDh, phong_exp);
			}

		}
	}
	prd_radiance.result = color;
}


//
// Set pixel to solid color upon failure
//
RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_color( bad_color );
}
