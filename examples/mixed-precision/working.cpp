#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "openacc.h"
#include "cuda_fp16.h"
#include "stdio.h"
#include <cuda_runtime.h>

struct dataobj
{
  void *restrict data;
  int * size;
  int * npsize;
  int * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
} ;

struct profiler
{
  double section0;
  double section1;
  double section2;
} ;

extern "C" int Forward(struct dataobj *restrict damp_vec, const float dt, const float o_x, const float o_y, const float o_z, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler * timers);
__device__ __half __hmul(__half a, __half b);

int Forward(struct dataobj *restrict damp_vec, const float dt, const float o_x, const float o_y, const float o_z, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler * timers)
{
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  #pragma acc enter data copyin(rec[0:rec_vec->size[0]][0:rec_vec->size[1]])
  #pragma acc enter data copyin(u[0:u_vec->size[0]][0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])
  #pragma acc enter data copyin(damp[0:damp_vec->size[0]][0:damp_vec->size[1]][0:damp_vec->size[2]])
  #pragma acc enter data copyin(rec_coords[0:rec_coords_vec->size[0]][0:rec_coords_vec->size[1]])
  #pragma acc enter data copyin(src[0:src_vec->size[0]][0:src_vec->size[1]])
  #pragma acc enter data copyin(src_coords[0:src_coords_vec->size[0]][0:src_coords_vec->size[1]])
  #pragma acc enter data copyin(vp[0:vp_vec->size[0]][0:vp_vec->size[1]][0:vp_vec->size[2]])
  
  __half a = 0.15;
  __half b = 0.15;
  float c = 0.25;
  __half c16 = __half2float(c);
  __half c16mul = __hmul(c16, a);
  
  
  //printf("%f", c);  
  for (int time = time_m, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
    #pragma acc parallel loop collapse(3)
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        for (int z = z_m; z <= z_M; z += 1)
        {
          float r15 = 1.0/dt;
          float r14 = 1.0/(dt*dt);
          float r13 = 1.0/(vp[x + 4][y + 4][z + 4]*vp[x + 4][y + 4][z + 4]);
          u[t2][x + 4][y + 4][z + 4] = (r13*(-r14*(-2.0F*u[t0][x + 4][y + 4][z + 4] + u[t1][x + 4][y + 4][z + 4])) + r15*(damp[x + 1][y + 1][z + 1]*u[t0][x + 4][y + 4][z + 4]) - 3.70370379e-4F*(u[t0][x + 2][y + 4][z + 4] + u[t0][x + 4][y + 2][z + 4] + u[t0][x + 4][y + 4][z + 2] + u[t0][x + 4][y + 4][z + 6] + u[t0][x + 4][y + 6][z + 4] + u[t0][x + 6][y + 4][z + 4]) + 5.92592607e-3F*(u[t0][x + 3][y + 4][z + 4] + u[t0][x + 4][y + 3][z + 4] + u[t0][x + 4][y + 4][z + 3] + u[t0][x + 4][y + 4][z + 5] + u[t0][x + 4][y + 5][z + 4] + u[t0][x + 5][y + 4][z + 4]) - 3.33333341e-2F*u[t0][x + 4][y + 4][z + 4])/(r13*r14 + r15*damp[x + 1][y + 1][z + 1]);
        }
      }
    }
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    /* Begin section1 */
    #pragma acc parallel loop collapse(1)
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
    {
      float posx = -o_x + src_coords[p_src][0];
      float posy = -o_y + src_coords[p_src][1];
      float posz = -o_z + src_coords[p_src][2];
      int ii_src_0 = (int)(floor(6.66667e-2*posx));
      int ii_src_1 = (int)(floor(6.66667e-2*posy));
      int ii_src_2 = (int)(floor(6.66667e-2*posz));
      int ii_src_3 = (int)(floor(6.66667e-2*posz)) + 1;
      int ii_src_4 = (int)(floor(6.66667e-2*posy)) + 1;
      int ii_src_5 = (int)(floor(6.66667e-2*posx)) + 1;
      float px = (float)(posx - 1.5e+1F*(int)(floor(6.66667e-2F*posx)));
      float py = (float)(posy - 1.5e+1F*(int)(floor(6.66667e-2F*posy)));
      float pz = (float)(posz - 1.5e+1F*(int)(floor(6.66667e-2F*posz)));
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
      {
        float r0 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4]*vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4])*(-2.96296e-4F*px*py*pz + 4.44445e-3F*px*py + 4.44445e-3F*px*pz - 6.66667e-2F*px + 4.44445e-3F*py*pz - 6.66667e-2F*py - 6.66667e-2F*pz + 1)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4] += r0;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
      {
        float r1 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4]*vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4])*(2.96296e-4F*px*py*pz - 4.44445e-3F*px*pz - 4.44445e-3F*py*pz + 6.66667e-2F*pz)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4] += r1;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        float r2 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4]*vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4])*(2.96296e-4F*px*py*pz - 4.44445e-3F*px*py - 4.44445e-3F*py*pz + 6.66667e-2F*py)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4] += r2;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        float r3 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4]*vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4])*(-2.96296e-4F*px*py*pz + 4.44445e-3F*py*pz)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4] += r3;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r4 = (dt*dt)*(vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4]*vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4])*(2.96296e-4F*px*py*pz - 4.44445e-3F*px*py - 4.44445e-3F*px*pz + 6.66667e-2F*px)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4] += r4;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r5 = (dt*dt)*(vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4]*vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4])*(-2.96296e-4F*px*py*pz + 4.44445e-3F*px*pz)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4] += r5;
      }
      if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r6 = (dt*dt)*(vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4]*vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4])*(-2.96296e-4F*px*py*pz + 4.44445e-3F*px*py)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4] += r6;
      }
      if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r7 = 2.96296e-4F*px*py*pz*(dt*dt)*(vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4]*vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4])*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4] += r7;
      }
    }
    /* End section1 */
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec-start_section1.tv_sec)+(double)(end_section1.tv_usec-start_section1.tv_usec)/1000000;
    struct timeval start_section2, end_section2;
    gettimeofday(&start_section2, NULL);
    /* Begin section2 */
    #pragma acc parallel loop collapse(1)
    for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
    {
      float posx = -o_x + rec_coords[p_rec][0];
      float posy = -o_y + rec_coords[p_rec][1];
      float posz = -o_z + rec_coords[p_rec][2];
      int ii_rec_0 = (int)(floor(6.66667e-2*posx));
      int ii_rec_1 = (int)(floor(6.66667e-2*posy));
      int ii_rec_2 = (int)(floor(6.66667e-2*posz));
      int ii_rec_3 = (int)(floor(6.66667e-2*posz)) + 1;
      int ii_rec_4 = (int)(floor(6.66667e-2*posy)) + 1;
      int ii_rec_5 = (int)(floor(6.66667e-2*posx)) + 1;
      float px = (float)(posx - 1.5e+1F*(int)(floor(6.66667e-2F*posx)));
      float py = (float)(posy - 1.5e+1F*(int)(floor(6.66667e-2F*posy)));
      float pz = (float)(posz - 1.5e+1F*(int)(floor(6.66667e-2F*posz)));
      float sum = 0.0F;
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1)
      {
        sum += (-2.96296e-4F*px*py*pz + 4.44445e-3F*px*py + 4.44445e-3F*px*pz - 6.66667e-2F*px + 4.44445e-3F*py*pz - 6.66667e-2F*py - 6.66667e-2F*pz + 1)*u[t0][ii_rec_0 + 4][ii_rec_1 + 4][ii_rec_2 + 4];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1)
      {
        sum += (2.96296e-4F*px*py*pz - 4.44445e-3F*px*pz - 4.44445e-3F*py*pz + 6.66667e-2F*pz)*u[t0][ii_rec_0 + 4][ii_rec_1 + 4][ii_rec_3 + 4];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1)
      {
        sum += (2.96296e-4F*px*py*pz - 4.44445e-3F*px*py - 4.44445e-3F*py*pz + 6.66667e-2F*py)*u[t0][ii_rec_0 + 4][ii_rec_4 + 4][ii_rec_2 + 4];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1)
      {
        sum += (-2.96296e-4F*px*py*pz + 4.44445e-3F*py*pz)*u[t0][ii_rec_0 + 4][ii_rec_4 + 4][ii_rec_3 + 4];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += (2.96296e-4F*px*py*pz - 4.44445e-3F*px*py - 4.44445e-3F*px*pz + 6.66667e-2F*px)*u[t0][ii_rec_5 + 4][ii_rec_1 + 4][ii_rec_2 + 4];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += (-2.96296e-4F*px*py*pz + 4.44445e-3F*px*pz)*u[t0][ii_rec_5 + 4][ii_rec_1 + 4][ii_rec_3 + 4];
      }
      if (ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += (-2.96296e-4F*px*py*pz + 4.44445e-3F*px*py)*u[t0][ii_rec_5 + 4][ii_rec_4 + 4][ii_rec_2 + 4];
      }
      if (ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += 2.96296e-4F*px*py*pz*u[t0][ii_rec_5 + 4][ii_rec_4 + 4][ii_rec_3 + 4];
      }
      rec[time][p_rec] = sum;
    }
    /* End section2 */
    gettimeofday(&end_section2, NULL);
    timers->section2 += (double)(end_section2.tv_sec-start_section2.tv_sec)+(double)(end_section2.tv_usec-start_section2.tv_usec)/1000000;
  }

  #pragma acc exit data copyout(rec[0:rec_vec->size[0]][0:rec_vec->size[1]])
  #pragma acc exit data delete(rec[0:rec_vec->size[0]][0:rec_vec->size[1]])
  #pragma acc exit data copyout(u[0:u_vec->size[0]][0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])
  #pragma acc exit data delete(u[0:u_vec->size[0]][0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])
  #pragma acc exit data delete(damp[0:damp_vec->size[0]][0:damp_vec->size[1]][0:damp_vec->size[2]])
  #pragma acc exit data delete(rec_coords[0:rec_coords_vec->size[0]][0:rec_coords_vec->size[1]])
  #pragma acc exit data delete(src[0:src_vec->size[0]][0:src_vec->size[1]])
  #pragma acc exit data delete(src_coords[0:src_coords_vec->size[0]][0:src_coords_vec->size[1]])
  #pragma acc exit data delete(vp[0:vp_vec->size[0]][0:vp_vec->size[1]][0:vp_vec->size[2]])
  return 0;
}
/* Backdoor edit at Tue Dec  8 17:14:27 2020*/ 
/* Backdoor edit at Tue Dec  8 17:15:01 2020*/ 
/* Backdoor edit at Tue Dec  8 17:16:36 2020*/ 
/* Backdoor edit at Tue Dec  8 17:17:07 2020*/ 
/* Backdoor edit at Tue Dec  8 17:17:34 2020*/ 
/* Backdoor edit at Tue Dec  8 17:17:55 2020*/ 
/* Backdoor edit at Tue Dec  8 17:20:06 2020*/ 
/* Backdoor edit at Tue Dec  8 17:22:08 2020*/ 
/* Backdoor edit at Tue Dec  8 17:26:17 2020*/ 
/* Backdoor edit at Tue Dec  8 17:27:53 2020*/ 
/* Backdoor edit at Tue Dec  8 17:29:23 2020*/ 
/* Backdoor edit at Tue Dec  8 17:33:14 2020*/ 
/* Backdoor edit at Tue Dec  8 17:33:46 2020*/ 
/* Backdoor edit at Tue Dec  8 17:37:12 2020*/ 
/* Backdoor edit at Tue Dec  8 17:38:22 2020*/ 
/* Backdoor edit at Tue Dec  8 17:39:01 2020*/ 
/* Backdoor edit at Tue Dec  8 17:40:31 2020*/ 
/* Backdoor edit at Tue Dec  8 20:06:55 2020*/ 
/* Backdoor edit at Tue Dec  8 20:07:57 2020*/ 
/* Backdoor edit at Tue Dec  8 20:15:07 2020*/ 
/* Backdoor edit at Tue Dec  8 20:17:22 2020*/ 
/* Backdoor edit at Tue Dec  8 20:20:23 2020*/ 
