/* 
 /*Render program functions, modified from work by NVIDIA
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

#define SAMPLE_ITERS 4.f

#define SAMPLE_ITERS_RECIP (1.f / SAMPLE_ITERS)

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

//rtDeclareVariable(float, dist, , );

RT_PROGRAM void thin_lens_camera()
{
	// Get ray direction of eye to image plane in the same way as before
	size_t2 screen = output_buffer.size();

	float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	// If lens_radius is 0, treat as pinhole
	if(lens_rad == 0)
	{
		optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon);

		PerRayData_radiance prd;
		prd.importance = 1.f;
		prd.depth = 0;

		rtTrace(top_object, ray, prd);

		output_buffer[launch_index] = make_color(prd.result);
	
	}
	else
	{
		///*

		//	Any ray that enters parallel to the axis on one side of the lens proceeds
		//	towards the focal point f on the other side

		//	Any ray that arrives at the lens after passing through the focal point on
		//	the front side comes out paralle to taxis on other side

		//	Any ray that passes through center of lens will not change direction

		//	Relation between distance s and image distance D' (Thin Lens Formula):

		//	(1/D) + (1/D') = (1/f);

		//	-> (1/D') = (1/f) - (1/D);

		//	-> D' = 1/((1/f) - (1/D));

		//	-> D' = (f * D) / (f + D)
		//*/

		//// Get D' value from distance and focal length
		//float dist_prime = (f_length * dist) / (f_length + dist);


		float3 result_color = make_float3(0.f, 0.f, 0.f);

		for (int i = 0; i < SAMPLE_ITERS; i++)
		{
			// 1) Sample point on lens
			unsigned seed = tea<2>(d.x, d.y);
			float2 sample_point = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

			//		float2 sample_point = d;

			//float2 sample_point = make_float2((d.x + 1.f) * 0.5f, (d.y + 1.f) * 0.5f);


			float2 disc_point =
				concentric_sample_disk(sample_point);


			float2 p_lens = lens_rad * disc_point;


			// 2) Compute point on plane of focus

			float ft = f_length / ray_direction.z;

			float3 pFocus = ray_origin + ray_direction * ft;

			// 3) Update ray for effect on lens

			ray_origin = make_float3(p_lens.x, p_lens.y, 0.f);
			ray_direction = normalize(pFocus - ray_origin);

			optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon);


			PerRayData_radiance prd;
			prd.importance = 1.f;
			prd.depth = 0;

			rtTrace(top_object, ray, prd);
			result_color += prd.result;
		}
		result_color *= SAMPLE_ITERS_RECIP;

		output_buffer[launch_index] = make_color(result_color);//make_color(prd.result);
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


//
// Dielectric surface shader
//
rtDeclareVariable(float3,       cutoff_color, , );
rtDeclareVariable(float,        fresnel_exponent, , );
rtDeclareVariable(float,        fresnel_minimum, , );
rtDeclareVariable(float,        fresnel_maximum, , );
rtDeclareVariable(float,        refraction_index, , );
rtDeclareVariable(int,          refraction_maxdepth, , );
rtDeclareVariable(int,          reflection_maxdepth, , );
rtDeclareVariable(float3,       refraction_color, , );
rtDeclareVariable(float3,       reflection_color, , );
rtDeclareVariable(float3,       extinction_constant, , );
rtDeclareVariable(float, importance_cutoff, , );
rtDeclareVariable(int, max_depth, , );

RT_PROGRAM void glass_closest_hit_radiance()
{
  // intersection vectors
  const float3 h = ray.origin + t_hit * ray.direction;            // hitpoint
  const float3 n = 
	  normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal
  const float3 i = ray.direction;                           // incident direction

  float reflection = 1.0f;
  float3 result = make_float3(0.0f);

  float3 beer_attenuation;
  if(dot(n, ray.direction) > 0){
    // Beer's law attenuation
    beer_attenuation = exp(extinction_constant * t_hit);
  } else {
    beer_attenuation = make_float3(1);
  }

  // refraction
  if (prd_radiance.depth < min(refraction_maxdepth, max_depth))
  {
    float3 t;											// transmission direction
    if ( refract(t, i, n, refraction_index) )
    {

      // check for external or internal reflection
      float cos_theta = dot(i, n);
      if (cos_theta < 0.0f)
        cos_theta = -cos_theta;
      else
        cos_theta = dot(t, n);

      reflection = fresnel_schlick(cos_theta,
		  fresnel_exponent,
		  fresnel_minimum,
		  fresnel_maximum);

      float importance = 
		  prd_radiance.importance 
		  * (1.0f-reflection) 
		  * optix::luminance( refraction_color * beer_attenuation );
      if ( importance > importance_cutoff ) {
        optix::Ray ray( h, t, radiance_ray_type, scene_epsilon );
        PerRayData_radiance refr_prd;
        refr_prd.depth = prd_radiance.depth+1;
        refr_prd.importance = importance;

        rtTrace( top_object, ray, refr_prd );
        result += (1.0f - reflection) * refraction_color * refr_prd.result;
      } else {
        result += (1.0f - reflection) * refraction_color * cutoff_color;
      }
    }
    // else TIR
  }

  // reflection
  if (prd_radiance.depth < min(reflection_maxdepth, max_depth))
  {
    float3 r = reflect(i, n);

    float importance = prd_radiance.importance 
		* reflection 
		* optix::luminance( reflection_color * beer_attenuation );
    if ( importance > importance_cutoff ) {
      optix::Ray ray( h, r, radiance_ray_type, scene_epsilon );
      PerRayData_radiance refl_prd;
      refl_prd.depth = prd_radiance.depth+1;
      refl_prd.importance = importance;

      rtTrace( top_object, ray, refl_prd );
      result += reflection * reflection_color * refl_prd.result;
    } else {
      result += reflection * reflection_color * cutoff_color;
    }
  }

  result = result * beer_attenuation;

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
rtBuffer<BasicLight>        lights;
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
