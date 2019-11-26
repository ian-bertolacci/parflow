#ifndef _PF_OMPLOOPS_DEBUG_H
#define _PF_OMPLOOPS_DEBUG_H

#define DEBUG_BoxLoopI1(locals,                                         \
                        i, j, k,                                        \
                        ix, iy, iz, nx, ny, nz,                         \
                        i1, nx1, ny1, nz1, sx1, sy1, sz1,               \
                        body)                                           \
	{                                                                     \
		DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    int debug_i1 = i1;                                                  \
    for (k = iz; k < iz + nz; k++)                                      \
    {                                                                   \
      for (j = iy; j < iy + ny; j++)                                    \
      {                                                                 \
        for (i = ix; i < ix + nx; i++)                                  \
        {                                                               \
          debug_i1 = INC_IDX((i - ix), (j - iy), (k - iz),              \
                       nx, ny, sx1, PV_jinc_1, PV_kinc_1);              \
          if (debug_i1 != i1)                                           \
          {                                                             \
            fprintf(stderr, "\nError: %d %d %d | %d vs %d at %s %d\n",  \
                    i, j, k, debug_i1, i1, __FILE__, __LINE__);         \
            exit(-1);                                                   \
          }                                                             \
          body;                                                         \
          i1 += sx1;                                                    \
        }                                                               \
        i1 += PV_jinc_1;                                                \
      }                                                                 \
      i1 += PV_kinc_1;                                                  \
    }                                                                   \
	}

#define DEBUG_BoxLoopI2(locals,                           \
                        i, j, k,                          \
                        ix, iy, iz, nx, ny, nz,           \
                        i1, nx1, ny1, nz1, sx1, sy1, sz1, \
                        i2, nx2, ny2, nz2, sx2, sy2, sz2, \
                        body)                             \
  {                                                                     \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2); \
    int debug_i1 = i1;                                                  \
    int debug_i2 = i2;                                                  \
    for (k = iz; k < iz + nz; k++)                                      \
    {                                                                   \
      for (j = iy; j < iy + ny; j++)                                    \
      {                                                                 \
        for (i = ix; i < ix + nx; i++)                                  \
        {                                                               \
          debug_i1 = INC_IDX((i-ix), (j-iy), (k-iz),                    \
                             nx, ny, sx1, PV_jinc_1, PV_kinc_1);        \
          debug_i2 = INC_IDX((i-ix), (j-iy), (k-iz),                    \
                             nx, ny, sx2, PV_jinc_2, PV_kinc_2);        \
          if (debug_i1 != i1 || debug_i2 != i2)                         \
          {                                                             \
            fprintf(stderr, "\nError: %d %d | %d %d at %s %d\n\n",      \
                    i, j, k, debug_i1, debug_i2, i1, i2, __FILE__, __LINE__); \
            exit(-1);                                                   \
          }                                                             \
          body;                                                         \
          i1 += sx1;                                                    \
          i2 += sx2;                                                    \
        }                                                               \
        i1 += PV_jinc_1;                                                \
        i2 += PV_jinc_2;                                                \
      }                                                                 \
      i1 += PV_kinc_1;                                                  \
      i2 += PV_kinc_2;                                                  \
    }                                                                   \
  }

#define DEBUG_BoxLoopI3(locals,                                         \
                   i, j, k,                                             \
                   ix, iy, iz, nx, ny, nz,                              \
                   i1, nx1, ny1, nz1, sx1, sy1, sz1,                    \
                   i2, nx2, ny2, nz2, sx2, sy2, sz2,                    \
                   i3, nx3, ny3, nz3, sx3, sy3, sz3,                    \
                   body)                                                \
  {                                                                     \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2); \
    DeclareInc(PV_jinc_3, PV_kinc_3, nx, ny, nz, nx3, ny3, nz3, sx3, sy3, sz3); \
    int debug_i1 = i1;                                                  \
    int debug_i2 = i2;                                                  \
    int debug_i3 = i3;                                                  \
    for (k = iz; k < iz + nz; k++)                                      \
    {                                                                   \
      for (j = iy; j < iy + ny; j++)                                    \
      {                                                                 \
        for (i = ix; i < ix + nx; i++)                                  \
        {                                                               \
          debug_i1 = INC_IDX((i - ix), (j - iy), (k - iz),              \
                             nx, ny, sx1, PV_jinc_1, PV_kinc_1);        \
          debug_i2 = INC_IDX((i - ix), (j - iy), (k - iz),              \
                             nx, ny, sx2, PV_jinc_2, PV_kinc_2);        \
          debug_i3 = INC_IDX((i - ix), (j - iy), (k - iz),              \
                             nx, ny, sx3, PV_jinc_3, PV_kinc_3);        \
          if (debug_i1 != i1 || debug_i2 != i2 || debug_i3 != i3)       \
          {                                                             \
            fprintf(stderr,                                             \
                    "\nError: %d %d %d starting %d %d %d | %d %d %d vs %d %d %d\n",\
                    i, j, k, ix, iy, iz,                                \
                    debug_i1, debug_i2, debug_i3,                       \
                    i1, i2, i3);                                        \
            fprintf(stderr,"At %s %d\n", __FILE__, __LINE__);           \
            fprintf(stderr,"Strides: %d %d %d\n", sx1, sx2, sx3);       \
            fprintf(stderr,"Incs: %d %d %d | %d %d %d\n",               \
                    PV_jinc_1, PV_jinc_2, PV_jinc_3,                    \
                    PV_kinc_1, PV_kinc_2, PV_kinc_3);                   \
            fprintf(stderr,"Size: %d %d %d\n", nx, ny, nz);             \
            exit(-1);                                                   \
          }                                                             \
                                                                        \
          body;                                                         \
          i1 += sx1;                                                    \
          i2 += sx2;                                                    \
          i3 += sx3;                                                    \
        }                                                               \
        i1 += PV_jinc_1;                                                \
        i2 += PV_jinc_2;                                                \
        i3 += PV_jinc_3;                                                \
      }                                                                 \
      i1 += PV_kinc_1;                                                  \
      i2 += PV_kinc_2;                                                  \
      i3 += PV_kinc_3;                                                  \
    }                                                                   \
  }


#endif // _PF_OMPLOOPS_DEBUG_H
