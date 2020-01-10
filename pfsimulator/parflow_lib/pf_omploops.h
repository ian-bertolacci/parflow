/* Macro redefinitions for OMP enabled loops */

#ifndef _PF_OMPLOOPS_H
#define _PF_OMPLOOPS_H

#include <omp.h>
#include <stdarg.h>


/* Utility macros for inserting OMP pragmas in macros */
#define EMPTY()
#define DEFER(x) x EMPTY()

#undef PRAGMA
#define PRAGMA(args) _Pragma( #args )

#undef PlusEquals
#define PlusEquals(a, b) OMP_PlusEquals(&(a), b)

#undef LOCALS
#define LOCALS(...) DEFER(_LOCALS)(__VA_ARGS__)
#define _LOCALS(...) ,__VA_ARGS__

#undef NO_LOCALS
#define NO_LOCALS DEFER(_NO_LOCALS)
#define _NO_LOCALS

/*
  Keeps the BoxLoop macros much tidier
  For GrGeomInLoop macros, the macro name will be used, not its expanded value
  i.e. In BoxLoopI# NoWait will expand to "for nowait", but in GrGeomInLoop it will
  be concatinated with _GrGeomInLoop resulting in NoWait_GrGeomInLoop
  This is necessary because going over the box clusters in a NewParallel section
  wraps the entire body of the macro inside the region for better performance,
  and so simply adjusting the pragma on the for loops is not sufficient.
*/
#define NewParallel parallel for
#define InParallel for
#define NoWait for nowait

/* Include other macro files AFTER we redefine the LOCALS pragmas */
#include "pf_omp_grgeom.h"


/* Helper Functions */
extern "C++"{

#pragma omp declare simd
  template<typename T>
  static inline void OMP_PlusEquals(T *arr_loc, T value)
  {
#pragma omp atomic update
    *arr_loc += value;
  }

#pragma omp declare simd
  template<typename T>
  static inline T _HarmonicMean(T a, T b)
  {
    return (a + b) ? (2.0 * a * b) / (a + b) : 0;
  }

#pragma omp declare simd
  template<typename T>
  static inline T _HarmonicMeanDZ(T a, T b, T c, T d)
  {
    return ((c * b) + (a * d)) ? ((c + d) * a * b) / ((b * c) + (a * d)) : 0;
  }

#pragma omp declare simd
  template<typename T>
  static inline T _UpstreamMean(T a, T b, T c, T d)
  {
    return (a - b) >= 0 ? c : d;
  }

#pragma omp delcare simd
  template<typename T>
  static inline T _ArithmeticMean(T a, T b)
  {
    return (0.5 * (a + b));
  }

#undef AtomicSet
#define AtomicSet(a, b) OMP_AtomicSet(&(a), b)
  template<typename T>
  static inline void OMP_AtomicSet(T *var, T val)
  {
#pragma omp atomic write
    *var = val;
  }

}

inline
void omp_Printf(const char *fmt, ...)
{
  #pragma omp master
  {
    va_list argp;
    va_start(argp, fmt);
    vfprintf(stderr, fmt, argp);
    va_end(argp);
  }
}

inline
void omp_AllPrintf(const char *fmt, ...)
{
  #pragma omp critical
  {
    va_list argp;
    va_start(argp, fmt);
    vfprintf(stderr, fmt, argp);
    va_end(argp);
  }
}

inline
void omp_AnyPrintf(const char *fmt, ...)
{
#pragma omp single nowait
  {
    va_list argp;
    va_start(argp, fmt);
    vfprintf(stderr, fmt, argp);
    va_end(argp);
  }
}


/*------------------------------------------------------------------------------
  BoxLoop Macro Redefinitions
  NOTE: The private pragmas will look odd.  This is because the NO_LOCALS macro
  expands to nothing, and the LOCALS(...) macro expands to ,__VA_ARGS__
  and so will insert the necessary comma itself.
  TODO: Come up with something that doesn't look like a crazy person wrote it
  -----------------------------------------------------------------------------*/

/* Util function to calculate the desired step in BoxLoopIX macros.
   Not tested with strides other than 1, but the math seems right.
   Pragma is to enable SIMD function calls from SIMD loops */
#pragma omp declare simd uniform(idx, nx, ny, sx, jinc, kinc)
inline int
INC_IDX(int idx, int i, int j, int k,
        int nx, int ny, int sx,
        int jinc, int kinc)
{
  return (k * kinc + (k * ny + j) * jinc +
          (k * ny * nx + j * nx + i) * sx) + idx;
}

#define MASTER(body) PRAGMA(omp master) body

#if 1

#undef InitMatrixLoop
#define InitMatrixLoop(s, stencil, ptr, mat_sub,                        \
                       i, j, k,                                         \
                       ix, iy, iz, nx, ny, nz,                          \
                       i1, nx1, ny1, nz1, sx1, sy1, sz1,                \
                       body)                                            \
  {                                                                     \
    for (s = 0; s < StencilSize(stencil); s++)                          \
    {                                                                   \
      ptr = SubmatrixElt(mat_sub, s, ix, iy, iz);                       \
      i1 = 0;                                                           \
      DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
      PRAGMA(omp for collapse(2) private(i, j, k, i1))                  \
        for (k = iz; k < iz + nz; k++)                                  \
        {                                                               \
          for (j = iy; j < iy + ny; j++)                                \
          {                                                             \
            PRAGMA(omp simd)                                            \
              for (i = ix; i < ix + nx; i++)                            \
              {                                                         \
                i1 = INC_IDX((i - ix), (j - iy), (k - iz),              \
                             nx, ny, sx1, PV_jinc_1, PV_kinc_1);        \
                body;                                                   \
              }                                                         \
          }                                                             \
        }                                                               \
    }                                                                   \
  }

#endif

#undef _BoxLoopI0
#define _BoxLoopI0(pragma, locals,                                \
                   i, j, k,                                       \
                   ix, iy, iz,                                    \
                   nx, ny, nz,                                    \
                   body)                                          \
  {                                                               \
    PRAGMA(omp pragma collapse(3) private(i, j, k locals))        \
      for (k = iz; k < iz + nz; k++)                              \
      {                                                           \
        for (j = iy; j < iy + ny; j++)                            \
        {                                                         \
          for (i = ix; i < ix + nx; i++)                          \
          {                                                       \
            body;                                                 \
          }                                                       \
        }                                                         \
      }                                                           \
  }

#undef _BoxLoopI1
#define _BoxLoopI1(pragma, locals,                                      \
                   i, j, k,                                             \
                   ix, iy, iz, nx, ny, nz,                              \
                   i1, nx1, ny1, nz1, sx1, sy1, sz1,                    \
                   body)                                                \
  {                                                                     \
    int i1_start = i1;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    PRAGMA(omp pragma collapse(3) private(i, j, k, i1 locals))          \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx1, PV_jinc_1, PV_kinc_1);            \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }

#undef _BoxLoopI2
#define _BoxLoopI2(pragma, locals,                                      \
                   i, j, k,                                             \
                   ix, iy, iz, nx, ny, nz,                              \
                   i1, nx1, ny1, nz1, sx1, sy1, sz1,                    \
                   i2, nx2, ny2, nz2, sx2, sy2, sz2,                    \
                   body)                                                \
  {                                                                     \
    int i1_start = i1;                                                  \
    int i2_start = i2;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2); \
    PRAGMA(omp pragma collapse(3) private(i, j, k, i1, i2 locals))      \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx1, PV_jinc_1, PV_kinc_1);            \
            i2 = INC_IDX(i2_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx2, PV_jinc_2, PV_kinc_2);            \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }

#undef _BoxLoopI2
#define _BoxLoopI2(pragma, locals,                                      \
                   i, j, k,                                             \
                   ix, iy, iz, nx, ny, nz,                              \
                   i1, nx1, ny1, nz1, sx1, sy1, sz1,                    \
                   i2, nx2, ny2, nz2, sx2, sy2, sz2,                    \
                   body)                                                \
  {                                                                     \
    int i1_start = i1;                                                  \
    int i2_start = i2;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2); \
    PRAGMA(omp pragma collapse(3) private(i, j, k, i1, i2 locals))      \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx1, PV_jinc_1, PV_kinc_1);            \
            i2 = INC_IDX(i2_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx2, PV_jinc_2, PV_kinc_2);            \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }

#undef _BoxLoopI3
#define _BoxLoopI3(pragma, locals,                                      \
                   i, j, k,                                             \
                   ix, iy, iz, nx, ny, nz,                              \
                   i1, nx1, ny1, nz1, sx1, sy1, sz1,                    \
                   i2, nx2, ny2, nz2, sx2, sy2, sz2,                    \
                   i3, nx3, ny3, nz3, sx3, sy3, sz3,                    \
                   body)                                                \
  {                                                                     \
    int i1_start = i1;                                                  \
    int i2_start = i2;                                                  \
    int i3_start = i3;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2); \
    DeclareInc(PV_jinc_3, PV_kinc_3, nx, ny, nz, nx3, ny3, nz3, sx3, sy3, sz3); \
    PRAGMA(omp pragma collapse(3) private(i, j, k, i1, i2, i3 locals))  \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx1, PV_jinc_1, PV_kinc_1);            \
            i2 = INC_IDX(i2_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx2, PV_jinc_2, PV_kinc_2);            \
            i3 = INC_IDX(i3_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx3, PV_jinc_3, PV_kinc_3);            \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }

#undef BoxLoopReduceI1
#define BoxLoopReduceI1(locals, sum,                                    \
                        i, j, k,                                        \
                        ix, iy, iz, nx, ny, nz,                         \
                        i1, nx1, ny1, nz1, sx1, sy1, sz1,               \
                        body)                                           \
  {                                                                     \
    int i1_start = i1;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    PRAGMA(omp parallel for reduction(+:sum) collapse(3) private(i, j, k, i1 locals)) \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx1, PV_jinc_1, PV_kinc_1);            \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }

#define __BoxLoopReduceI1(locals, sum,                                  \
                          i, j, k,                                      \
                          ix, iy, iz, nx, ny, nz,                       \
                          i1, nx1, ny1, nz1, sx1, sy1, sz1,             \
                          body)                                         \
  {                                                                     \
    int i1_start = i1;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    PRAGMA(omp for reduction(+:sum) collapse(3) private(i, j, k, i1 locals)) \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx1, PV_jinc_1, PV_kinc_1);            \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }

#undef BoxLoopReduceI2
#define BoxLoopReduceI2(locals, sum,                                    \
                        i, j, k,                                        \
                        ix, iy, iz, nx, ny, nz,                         \
                        i1, nx1, ny1, nz1, sx1, sy1, sz1,               \
                        i2, nx2, ny2, nz2, sx2, sy2, sz2,               \
                        body)                                           \
  {                                                                     \
    int i1_start = i1;                                                  \
    int i2_start = i2;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2); \
    PRAGMA(omp parallel for reduction(+:sum) collapse(3) private(i, j, k, i1, i2 locals)) \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx1, PV_jinc_1, PV_kinc_1);            \
            i2 = INC_IDX(i2_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx2, PV_jinc_2, PV_kinc_2);            \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }


#if 1
/*------------------------------------------------------------------------
 * BCPatchLoop Redefinitions
 *------------------------------------------------------------------------*/
#undef _BCStructPatchLoop
#define _BCStructPatchLoop(locals,                                      \
                           i, j, k, fdir, ival, bc_struct, ipatch, is, body) \
  {                                                                     \
    GrGeomSolid  *PV_gr_domain = BCStructGrDomain(bc_struct);           \
    int PV_patch_index = BCStructPatchIndex(bc_struct, ipatch);         \
    Subgrid      *PV_subgrid = BCStructSubgrid(bc_struct, is);          \
                                                                        \
    int PV_r = SubgridRX(PV_subgrid);                                   \
    int PV_ix = SubgridIX(PV_subgrid);                                  \
    int PV_iy = SubgridIY(PV_subgrid);                                  \
    int PV_iz = SubgridIZ(PV_subgrid);                                  \
    int PV_nx = SubgridNX(PV_subgrid);                                  \
    int PV_ny = SubgridNY(PV_subgrid);                                  \
    int PV_nz = SubgridNZ(PV_subgrid);                                  \
                                                                        \
    ival = 0;                                                           \
    if (PV_r == 0 && GrGeomSolidPatchBoxes(PV_gr_domain, PV_patch_index, GrGeomOctreeNumFaces -1)) \
    {                                                                   \
      _GrGeomPatchLoopBoxes(locals, i, j, k, fdir,                      \
                            PV_gr_domain, PV_patch_index, PV_r,         \
                            PV_ix, PV_iy, PV_iz, PV_nx, PV_ny, PV_nz,   \
                            body);                                      \
    }                                                                   \
    else                                                                \
    {                                                                   \
      GrGeomPatchLoop(i, j, k, fdir, PV_gr_domain, PV_patch_index,      \
                      PV_r, PV_ix, PV_iy, PV_iz, PV_nx, PV_ny, PV_nz,   \
      {                                                                 \
        body;                                                           \
        ival++;                                                         \
      });                                                               \
    }                                                                   \
  }


//#undef _GrGeomPatchLoopBoxes
#define _GrGeomPatchLoopBoxes(locals,                                   \
                              i, j, k, fdir, grgeom, patch_num,         \
                              r, ix, iy, iz, nx, ny, nz, body)          \
  {                                                                     \
                                                                        \
    PRAGMA(omp parallel private(fdir))                                  \
    {                                                                   \
      int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;               \
      int *PV_visiting = NULL;                                          \
      int PV_fdir[3];                                                   \
      fdir = PV_fdir;                                                   \
      for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)           \
      {                                                                 \
        switch (PV_f)                                                   \
        {                                                               \
          case GrGeomOctreeFaceL:                                       \
            fdir[0] = -1; fdir[1] = 0; fdir[2] = 0;                     \
            break;                                                      \
          case GrGeomOctreeFaceR:                                       \
            fdir[0] = 1; fdir[1] = 0; fdir[2] = 0;                      \
            break;                                                      \
          case GrGeomOctreeFaceD:                                       \
            fdir[0] = 0; fdir[1] = -1; fdir[2] = 0;                     \
            break;                                                      \
          case GrGeomOctreeFaceU:                                       \
            fdir[0] = 0; fdir[1] = 1; fdir[2] = 0;                      \
            break;                                                      \
          case GrGeomOctreeFaceB:                                       \
            fdir[0] = 0; fdir[1] = 0; fdir[2] = -1;                     \
            break;                                                      \
          case GrGeomOctreeFaceF:                                       \
            fdir[0] = 0; fdir[1] = 0; fdir[2] = 1;                      \
            break;                                                      \
          default:                                                      \
            fdir[0] = -9999; fdir[1] = -9999; fdir[2] = -99999;         \
            break;                                                      \
        }                                                               \
                                                                        \
        BoxArray* boxes = GrGeomSolidPatchBoxes(grgeom, patch_num, PV_f); \
        for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)    \
        {                                                               \
          Box box = BoxArrayGetBox(boxes, PV_box);                      \
          /* find octree and region intersection */                     \
          PV_ixl = pfmax(ix, box.lo[0]);                                \
          PV_iyl = pfmax(iy, box.lo[1]);                                \
          PV_izl = pfmax(iz, box.lo[2]);                                \
          PV_ixu = pfmin((ix + nx - 1), box.up[0]);                     \
          PV_iyu = pfmin((iy + ny - 1), box.up[1]);                     \
          PV_izu = pfmin((iz + nz - 1), box.up[2]);                     \
                                                                        \
          int PV_diff_x = PV_ixu - PV_ixl;                              \
          int PV_diff_y = PV_iyu - PV_iyl;                              \
          int PV_diff_z = PV_izu - PV_izl;                              \
          int x_scale = !!PV_diff_x;                                    \
          int y_scale = !!PV_diff_y;                                    \
          int z_scale = !!PV_diff_z;                                    \
          if (PV_diff_x * PV_diff_y * PV_diff_z != 0) {                 \
            fprintf(stderr, "ERROR: Diff not 0 at %s %d\n", __FILE__, __LINE__); \
            exit(-1);                                                   \
          }                                                             \
          PRAGMA(omp for collapse(3) private(i, j, k, ival locals))     \
            for (k = PV_izl; k <= PV_izu; k++)                          \
            {                                                           \
              for (j = PV_iyl; j <= PV_iyu; j++)                        \
              {                                                         \
                for (i = PV_ixl; i <= PV_ixu; i++)                      \
                {                                                       \
                  if (!z_scale) {                                       \
                    ival = (PV_diff_x * j + j + i);                     \
                  } else if (!y_scale) {                                \
                    ival = (PV_diff_x * k + k + i);                     \
                  } else {                                              \
                    ival = (PV_diff_y * k + k + j);                     \
                  }                                                     \
                  body;                                                 \
                }                                                       \
              }                                                         \
            }                                                           \
        }                                                               \
      }                                                                 \
    }                                                                   \
  }


#define __BCStructPatchLoop(locals,                                     \
                           i, j, k, fdir, ival, bc_struct, ipatch, is, body) \
  {                                                                     \
    GrGeomSolid  *PV_gr_domain = BCStructGrDomain(bc_struct);           \
    int PV_patch_index = BCStructPatchIndex(bc_struct, ipatch);         \
    Subgrid      *PV_subgrid = BCStructSubgrid(bc_struct, is);          \
                                                                        \
    int PV_r = SubgridRX(PV_subgrid);                                   \
    int PV_ix = SubgridIX(PV_subgrid);                                  \
    int PV_iy = SubgridIY(PV_subgrid);                                  \
    int PV_iz = SubgridIZ(PV_subgrid);                                  \
    int PV_nx = SubgridNX(PV_subgrid);                                  \
    int PV_ny = SubgridNY(PV_subgrid);                                  \
    int PV_nz = SubgridNZ(PV_subgrid);                                  \
                                                                        \
    ival = 0;                                                           \
    if (PV_r == 0 && GrGeomSolidPatchBoxes(PV_gr_domain, PV_patch_index, GrGeomOctreeNumFaces -1)) \
    {                                                                   \
      __GrGeomPatchLoopBoxes(locals, i, j, k, fdir,                     \
                            PV_gr_domain, PV_patch_index, PV_r,         \
                            PV_ix, PV_iy, PV_iz, PV_nx, PV_ny, PV_nz,   \
                            body);                                      \
    }                                                                   \
    else                                                                \
    {                                                                   \
      GrGeomPatchLoop(i, j, k, fdir, PV_gr_domain, PV_patch_index,      \
                      PV_r, PV_ix, PV_iy, PV_iz, PV_nx, PV_ny, PV_nz,   \
      {                                                                 \
        body;                                                           \
        ival++;                                                         \
      });                                                               \
    }                                                                   \
  }


#define __GrGeomPatchLoopBoxes(locals,																	\
                               i, j, k, fdir, grgeom, patch_num,        \
                               r, ix, iy, iz, nx, ny, nz, body)         \
  {                                                                     \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;                 \
    int *PV_visiting = NULL;                                            \
    int PV_fdir[3];                                                     \
    fdir = PV_fdir;                                                     \
    for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)             \
    {                                                                   \
      switch (PV_f)                                                     \
      {                                                                 \
        case GrGeomOctreeFaceL:                                         \
          fdir[0] = -1; fdir[1] = 0; fdir[2] = 0;                       \
          break;                                                        \
        case GrGeomOctreeFaceR:                                         \
          fdir[0] = 1; fdir[1] = 0; fdir[2] = 0;                        \
          break;                                                        \
        case GrGeomOctreeFaceD:                                         \
          fdir[0] = 0; fdir[1] = -1; fdir[2] = 0;                       \
          break;                                                        \
        case GrGeomOctreeFaceU:                                         \
          fdir[0] = 0; fdir[1] = 1; fdir[2] = 0;                        \
          break;                                                        \
        case GrGeomOctreeFaceB:                                         \
          fdir[0] = 0; fdir[1] = 0; fdir[2] = -1;                       \
          break;                                                        \
        case GrGeomOctreeFaceF:                                         \
          fdir[0] = 0; fdir[1] = 0; fdir[2] = 1;                        \
          break;                                                        \
        default:                                                        \
          fdir[0] = -9999; fdir[1] = -9999; fdir[2] = -99999;           \
          break;                                                        \
      }                                                                 \
                                                                        \
      BoxArray* boxes = GrGeomSolidPatchBoxes(grgeom, patch_num, PV_f); \
      for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)      \
      {                                                                 \
        Box box = BoxArrayGetBox(boxes, PV_box);                        \
        /* find octree and region intersection */                       \
        PV_ixl = pfmax(ix, box.lo[0]);                                  \
        PV_iyl = pfmax(iy, box.lo[1]);                                  \
        PV_izl = pfmax(iz, box.lo[2]);                                  \
        PV_ixu = pfmin((ix + nx - 1), box.up[0]);                       \
        PV_iyu = pfmin((iy + ny - 1), box.up[1]);                       \
        PV_izu = pfmin((iz + nz - 1), box.up[2]);                       \
                                                                        \
        int PV_diff_x = PV_ixu - PV_ixl;                                \
        int PV_diff_y = PV_iyu - PV_iyl;                                \
        int PV_diff_z = PV_izu - PV_izl;                                \
        int x_scale = !!PV_diff_x;                                      \
        int y_scale = !!PV_diff_y;                                      \
        int z_scale = !!PV_diff_z;                                      \
        if (PV_diff_x * PV_diff_y * PV_diff_z != 0) {                   \
          fprintf(stderr, "ERROR: Diff not 0 at %s %d\n", __FILE__, __LINE__); \
          exit(-1);                                                     \
        }                                                               \
        PRAGMA(omp for collapse(3) private(i, j, k, ival locals))       \
          for (k = PV_izl; k <= PV_izu; k++)                            \
          {                                                             \
            for (j = PV_iyl; j <= PV_iyu; j++)                          \
            {                                                           \
              for (i = PV_ixl; i <= PV_ixu; i++)                        \
              {                                                         \
                if (!z_scale) {                                         \
                  ival = (PV_diff_x * j + j + i);                       \
                } else if (!y_scale) {                                  \
                  ival = (PV_diff_x * k + k + i);                       \
                } else {                                                \
                  ival = (PV_diff_y * k + k + j);                       \
                }                                                       \
                body;                                                   \
              }                                                         \
            }                                                           \
          }                                                             \
      }                                                                 \
    }                                                                   \
	}

#endif // BCStruct #if

#endif // _PF_OMPLOOPS_H
