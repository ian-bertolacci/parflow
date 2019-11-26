/* Macro redefinitions for OMP enabled loops */

#ifndef _PF_OMPLOOPS_H
#define _PF_OMPLOOPS_H

#include <omp.h>

#undef PlusEquals
#define PlusEquals(a, b) OMP_PlusEquals(&(a), b)

extern "C++"{
  template<typename T>
  static inline void OMP_PlusEquals(T *arr_loc, T value)
  {
#pragma omp atomic
    *arr_loc += value;
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
#pragma omp declare simd
inline int
INC_IDX(int i, int j, int k,
        int nx, int ny, int sx,
        int jinc, int kinc)
{
  return (k * kinc + (k * ny + j) * jinc +
          (k * ny * nx + j * nx + i) * sx);
}


#undef _BoxLoopI0
#define _BoxLoopI0(locals,																				\
									 i, j, k,																				\
									 ix, iy, iz,																		\
									 nx, ny, nz,																		\
									 body)																					\
  {																																\
    PRAGMA(omp parallel for collapse(3) private(i, j, k locals))	\
			for (k = iz; k < iz + nz; k++)															\
			{																														\
				for (j = iy; j < iy + ny; j++)														\
				{																													\
					for (i = ix; i < ix + nx; i++)													\
					{																												\
						body;																									\
					}																												\
				}																													\
			}																														\
  }

#undef _BoxLoopI1
#define _BoxLoopI1(locals,                                              \
                   i, j, k,                                             \
                   ix, iy, iz, nx, ny, nz,                              \
                   i1, nx1, ny1, nz1, sx1, sy1, sz1,                    \
                   body)                                                \
	{                                                                     \
		DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
		PRAGMA(omp parallel for num_threads(1) collapse(3) private(i, j, k, i1 locals)) \
			for (k = iz; k < iz + nz; k++)																		\
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i, j, k, nx, ny, sx1, PV_jinc_1, PV_kinc_1);   \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
	}

#if 0
#undef _BoxLoopI2
#define _BoxLoopI2(locals,                                              \
                   i, j, k,                                             \
                   ix, iy, iz, nx, ny, nz,                              \
                   i1, nx1, ny1, nz1, sx1, sy1, sz1,                    \
                   i2, nx2, ny2, nz2, sx2, sy2, sz2,                    \
                   body)                                                \
  {                                                                     \
		int temp1 = i1;																											\
		int temp2 = i2;																											\
		DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
		DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2); \
		for (k = iz; k < iz + nz; k++)																			\
		{																																		\
			for (j = iy; j < iy + ny; j++)																		\
			{																																	\
				for (i = ix; i < ix + nx; i++)																	\
				{																																\
					temp1 = INC_IDX(i, j, k, ny, nx, sx1, PV_jinc_1, PV_kinc_1);	\
					temp2 = INC_IDX(i, j, k, ny, nx, sx2, PV_jinc_2, PV_kinc_2);	\
					fprintf(stderr, "%d %d | %d %d | %d %d %d\n", temp1, temp2, i1, i2, i, j, k); \
					if (temp1 != i1 || temp2 != i2) {															\
						fprintf(stderr, "Error: %d - %d and %d - %d at %d %d %d\n",	\
										temp1, i1, temp2, i2, i, j, k);											\
						exit(0);																										\
					}																															\
					body;																													\
					i1 += sx1;																										\
					i2 += sx2;																										\
				}																																\
				i1 += PV_jinc_1;																								\
				i2 += PV_jinc_2;																								\
			}																																	\
			i1 += PV_kinc_1;																									\
			i2 += PV_kinc_2;																									\
		}																																		\
  }
#endif

#undef _BoxLoopI2
#define _BoxLoopI2(locals,                                              \
                   i, j, k,                                             \
                   ix, iy, iz, nx, ny, nz,                              \
                   i1, nx1, ny1, nz1, sx1, sy1, sz1,                    \
                   i2, nx2, ny2, nz2, sx2, sy2, sz2,                    \
                   body)                                                \
  {                                                                     \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2); \
    PRAGMA(omp parallel for collapse(3) private(i, j, k, i1, i2 locals)) \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i, j, k, ny, nx, sx1, PV_jinc_1, PV_kinc_1);   \
            i2 = INC_IDX(i, j, k, ny, nx, sx2, PV_jinc_2, PV_kinc_2);   \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }


#undef _BoxLoopI3
#define _BoxLoopI3(locals,                                              \
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
    PRAGMA(omp parallel for collapse(3) private(i, j, k, i1, i2, i3 locals)) \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i, j, k, ny, nx, sx1, PV_jinc_1, PV_kinc_1);   \
            i2 = INC_IDX(i, j, k, ny, nx, sx2, PV_jinc_2, PV_kinc_2);   \
            i3 = INC_IDX(i, j, k, ny, nx, sx3, PV_jinc_3, PV_kinc_3);   \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }


#undef BoxLoopReduceI1
#define BoxLoopReduceI1(sum,                                            \
                        i, j, k,                                        \
                        ix, iy, iz, nx, ny, nz,                         \
                        i1, nx1, ny1, nz1, sx1, sy1, sz1,               \
                        body)                                           \
  {                                                                     \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    PRAGMA(omp parallel for reduction(+:sum) collapse(3) private(i, j, k, i1)) \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i, j, k, nx, ny, sx1, PV_jinc_1, PV_kinc_1);   \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }

/*------------------------------------------------------------------------
 * Clustering Box Loop Redefinitions
 *------------------------------------------------------------------------*/

/*------------------------------------------------------------------------
 * GrGeomInLoop Redefinitions
 *------------------------------------------------------------------------*/
#undef _GrGeomInLoop
#define _GrGeomInLoop(locals, i, j, k, grgeom, r, ix, iy, iz, nx, ny, nz, body) \
	{																																			\
		if (r == 0 && GrGeomSolidInteriorBoxes(grgeom))											\
		{																																		\
			_GrGeomInLoopBoxes(locals,																				\
                         i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body); \
		}																																		\
		else																																\
		{																																		\
			GrGeomOctree  *PV_node;																						\
			double PV_ref = pow(2.0, r);																			\
																																				\
			i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;										\
			j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;										\
			k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;										\
			GrGeomOctreeInteriorNodeLoop(i, j, k, PV_node,										\
																	 GrGeomSolidData(grgeom),							\
																	 GrGeomSolidOctreeBGLevel(grgeom) + r, \
																	 ix, iy, iz, nx, ny, nz,							\
																	 TRUE,																\
																	 body);																\
		}																																		\
	}

//#undef GrGeomInLoopBoxes
#define _GrGeomInLoopBoxes(locals, i, j, k,                         \
													 grgeom, ix, iy, iz,                      \
													 nx, ny, nz, body)                        \
  {                                                                 \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;             \
    int *PV_visiting = NULL;                                        \
    BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);             \
		PRAGMA(omp parallel)                                            \
		{                                                               \
			for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)  \
			{                                                             \
				Box box = BoxArrayGetBox(boxes, PV_box);                    \
				/* find octree and region intersection */                   \
				PV_ixl = pfmax(ix, box.lo[0]);                              \
				PV_iyl = pfmax(iy, box.lo[1]);                              \
				PV_izl = pfmax(iz, box.lo[2]);                              \
				PV_ixu = pfmin((ix + nx - 1), box.up[0]);                   \
				PV_iyu = pfmin((iy + ny - 1), box.up[1]);                   \
				PV_izu = pfmin((iz + nz - 1), box.up[2]);                   \
                                                                    \
				PRAGMA(omp for collapse(3) private(i, j, k locals))         \
					for (k = PV_izl; k <= PV_izu; k++)                        \
					{                                                         \
						for (j = PV_iyl; j <= PV_iyu; j++)                      \
						{                                                       \
							for (i = PV_ixl; i <= PV_ixu; i++)                    \
							{                                                     \
								body;                                               \
							}                                                     \
						}                                                       \
					}                                                         \
			}                                                             \
		}                                                               \
	}


/*------------------------------------------------------------------------
 * BCPatchLoop Redefinitions
 *------------------------------------------------------------------------*/
#undef _BCStructPatchLoop
#define _BCStructPatchLoop(locals,																			\
													 i, j, k, fdir, ival, bc_struct, ipatch, is, body) \
  {																																			\
		GrGeomSolid  *PV_gr_domain = BCStructGrDomain(bc_struct);						\
		int PV_patch_index = BCStructPatchIndex(bc_struct, ipatch);					\
		Subgrid      *PV_subgrid = BCStructSubgrid(bc_struct, is);					\
																																				\
		int PV_r = SubgridRX(PV_subgrid);																		\
		int PV_ix = SubgridIX(PV_subgrid);																	\
		int PV_iy = SubgridIY(PV_subgrid);																	\
		int PV_iz = SubgridIZ(PV_subgrid);																	\
		int PV_nx = SubgridNX(PV_subgrid);																	\
		int PV_ny = SubgridNY(PV_subgrid);																	\
		int PV_nz = SubgridNZ(PV_subgrid);																	\
																																				\
		ival = 0;																														\
		if (PV_r == 0 && GrGeomSolidPatchBoxes(PV_gr_domain, PV_patch_index, GrGeomOctreeNumFaces -1)) \
		{																																		\
			_GrGeomPatchLoopBoxes(locals, i, j, k, fdir,                      \
                            PV_gr_domain, PV_patch_index, PV_r,         \
                            PV_ix, PV_iy, PV_iz, PV_nx, PV_ny, PV_nz,   \
                            body);                                      \
		}																																		\
		else																																\
		{																																		\
			GrGeomPatchLoop(i, j, k, fdir, PV_gr_domain, PV_patch_index,			\
											PV_r, PV_ix, PV_iy, PV_iz, PV_nx, PV_ny, PV_nz,		\
			{																																	\
				body;																														\
				ival++;																													\
			});																																\
		}																																		\
	}

//#undef _GrGeomPatchLoopBoxes
#define _GrGeomPatchLoopBoxes(locals,                                   \
                              i, j, k, fdir, grgeom, patch_num,         \
                              r, ix, iy, iz, nx, ny, nz, body)          \
	{																																			\
    int PV_fdir[3];																											\
    																																		\
    fdir = PV_fdir;																											\
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;									\
    int *PV_visiting = NULL;																						\
    PRAGMA(omp parallel)                                                \
    {                                                                   \
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

/*
  int k1 = k * z_scale;
  int j1 = j * y_scale;
  int i1 = i * x_scale;
  if (z_scale == 0) {
  ival = (PV_diff_x * j + j + i);
} else if (y_scale == 0) {
ival = (PV_diff_x * k + k + i);
} else {
ival = (PV_diff_y * k + k + j)
}
((k * PV_diff_y + k + j) * z_scale) + ((k * PV_diff_x + k + i) * z_scale)
  PV_diff_y * k + k + j
 */
#endif // _PF_OMPLOOPS_H
