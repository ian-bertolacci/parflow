/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
 *  LLC. Produced at the Lawrence Livermore National Laboratory. Written
 *  by the Parflow Team (see the CONTRIBUTORS file)
 *  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
 *
 *  This file is part of Parflow. For details, see
 *  http://www.llnl.gov/casc/parflow
 *
 *  Please read the COPYRIGHT file or Our Notice and the LICENSE file
 *  for the GNU Lesser General Public License.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License (as published
 *  by the Free Software Foundation) version 2.1 dated February 1999.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
 *  and conditions of the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 *  USA
 **********************************************************************EHEADER*/
/*****************************************************************************
* Geometry class structures and accessors
*
*****************************************************************************/

#ifndef _GR_GEOMETRY_HEADER
#define _GR_GEOMETRY_HEADER

#include "geometry.h"
#include "grgeom_octree.h"
#include "grgeom_list.h"
#include "index_space.h"
#include <omp.h>
#include <assert.h>

/*--------------------------------------------------------------------------
 * Miscellaneous structures:
 *--------------------------------------------------------------------------*/

typedef int GrGeomExtents[6];

typedef struct {
  GrGeomExtents  *extents;
  int size;
} GrGeomExtentArray;


/*--------------------------------------------------------------------------
 * Solid structures:
 *--------------------------------------------------------------------------*/

typedef struct {
  GrGeomOctree  *data;

  GrGeomOctree **patches;
  int num_patches;

  /* these fields are used to relate the background with the octree */
  int octree_bg_level;
  int octree_ix, octree_iy, octree_iz;

  /* Boxes for iteration */

  BoxArray* interior_boxes;
  BoxArray* surface_boxes[GrGeomOctreeNumFaces];
  BoxArray** patch_boxes[GrGeomOctreeNumFaces];
} GrGeomSolid;


/*--------------------------------------------------------------------------
 * Accessor macros:
 *--------------------------------------------------------------------------*/

#define GrGeomExtentsIXLower(extents)  ((extents)[0])
#define GrGeomExtentsIXUpper(extents)  ((extents)[1])
#define GrGeomExtentsIYLower(extents)  ((extents)[2])
#define GrGeomExtentsIYUpper(extents)  ((extents)[3])
#define GrGeomExtentsIZLower(extents)  ((extents)[4])
#define GrGeomExtentsIZUpper(extents)  ((extents)[5])

#define GrGeomExtentArrayExtents(ext_array)  ((ext_array)->extents)
#define GrGeomExtentArraySize(ext_array)     ((ext_array)->size)

#define GrGeomSolidData(solid)          ((solid)->data)
#define GrGeomSolidPatches(solid)       ((solid)->patches)
#define GrGeomSolidNumPatches(solid)    ((solid)->num_patches)
#define GrGeomSolidOctreeBGLevel(solid) ((solid)->octree_bg_level)
#define GrGeomSolidOctreeIX(solid)      ((solid)->octree_ix)
#define GrGeomSolidOctreeIY(solid)      ((solid)->octree_iy)
#define GrGeomSolidOctreeIZ(solid)      ((solid)->octree_iz)
#define GrGeomSolidPatch(solid, i)      ((solid)->patches[(i)])
#define GrGeomSolidInteriorBoxes(solid) ((solid)->interior_boxes)
#define GrGeomSolidSurfaceBoxes(solid, i)  ((solid)->surface_boxes[(i)])
#define GrGeomSolidPatchBoxes(solid, patch, i)  ((solid)->patch_boxes[(i)][(patch)])

/*==========================================================================
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the inside of a solid.
 *   Serial in all aspects.
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxes(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                 \
  {                                                                           \
    Box box = BoxArrayGetBox(boxes, PV_box);                                  \
    /* find octree and region intersection */                                 \
    int PV_ixl = pfmax(ix, box.lo[0]);                                        \
    int PV_iyl = pfmax(iy, box.lo[1]);                                        \
    int PV_izl = pfmax(iz, box.lo[2]);                                        \
    int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                             \
    int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                             \
    int PV_izu = pfmin((iz + nz - 1), box.up[2]);                             \
                                                                              \
    for(k = PV_izl; k <= PV_izu; k++)                                         \
      for(j =PV_iyl; j <= PV_iyu; j++)                                        \
        for(i = PV_ixl; i <= PV_ixu; i++)                                     \
        {                                                                     \
          body;                                                               \
        }                                                                     \
   }                                                                          \
}


/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel over boxes
 *   Parallelized over boxes using default schedule
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesParallelOverBoxes(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  PRAGMA_IN_MACRO_BODY( omp parallel for private(i,j,k) )    \
  for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                 \
  {                                                                           \
    Box box = BoxArrayGetBox(boxes, PV_box);                                  \
    /* find octree and region intersection */                                 \
    int PV_ixl = pfmax(ix, box.lo[0]);                                        \
    int PV_iyl = pfmax(iy, box.lo[1]);                                        \
    int PV_izl = pfmax(iz, box.lo[2]);                                        \
    int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                             \
    int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                             \
    int PV_izu = pfmin((iz + nz - 1), box.up[2]);                             \
                                                                              \
    for(k = PV_izl; k <= PV_izu; k++)                                         \
      for(j =PV_iyl; j <= PV_iyu; j++)                                        \
        for(i = PV_ixl; i <= PV_ixu; i++)                                     \
        {                                                                     \
          body;                                                               \
        }                                                                     \
   }                                                                          \
}

/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel over boxes
 *   Parallelized over boxes using schedule(dynamic) with default chunk size
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesParallelOverBoxesDynamicDefault(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  PRAGMA_IN_MACRO_BODY( omp parallel for private(i,j,k) schedule(dynamic))    \
  for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                 \
  {                                                                           \
    Box box = BoxArrayGetBox(boxes, PV_box);                                  \
    /* find octree and region intersection */                                 \
    int PV_ixl = pfmax(ix, box.lo[0]);                                        \
    int PV_iyl = pfmax(iy, box.lo[1]);                                        \
    int PV_izl = pfmax(iz, box.lo[2]);                                        \
    int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                             \
    int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                             \
    int PV_izu = pfmin((iz + nz - 1), box.up[2]);                             \
                                                                              \
    for(k = PV_izl; k <= PV_izu; k++)                                         \
      for(j =PV_iyl; j <= PV_iyu; j++)                                        \
        for(i = PV_ixl; i <= PV_ixu; i++)                                     \
        {                                                                     \
          body;                                                               \
        }                                                                     \
   }                                                                          \
}

/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel over boxes
 *   Parallelized over boxes using schedule(guided) with default chunk size
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesParallelOverBoxesDynamicGuided(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  PRAGMA_IN_MACRO_BODY( omp parallel for private(i,j,k) schedule(guided))     \
  for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                 \
  {                                                                           \
    Box box = BoxArrayGetBox(boxes, PV_box);                                  \
    /* find octree and region intersection */                                 \
    int PV_ixl = pfmax(ix, box.lo[0]);                                        \
    int PV_iyl = pfmax(iy, box.lo[1]);                                        \
    int PV_izl = pfmax(iz, box.lo[2]);                                        \
    int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                             \
    int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                             \
    int PV_izu = pfmin((iz + nz - 1), box.up[2]);                             \
                                                                              \
    for(k = PV_izl; k <= PV_izu; k++)                                         \
      for(j =PV_iyl; j <= PV_iyu; j++)                                        \
        for(i = PV_ixl; i <= PV_ixu; i++)                                     \
        {                                                                     \
          body;                                                               \
        }                                                                     \
   }                                                                          \
}

/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel in boxes
 *   Parallel in boxes, parallel on the Z loop (outermost loop)
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesParallelInBoxesOnZ(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  PRAGMA_IN_MACRO_BODY( omp parallel )                                        \
  {                                                                           \
    for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)               \
    {                                                                         \
      Box box = BoxArrayGetBox(boxes, PV_box);                                \
      /* find octree and region intersection */                               \
      int PV_ixl = pfmax(ix, box.lo[0]);                                      \
      int PV_iyl = pfmax(iy, box.lo[1]);                                      \
      int PV_izl = pfmax(iz, box.lo[2]);                                      \
      int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                           \
      int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                           \
      int PV_izu = pfmin((iz + nz - 1), box.up[2]);                           \
      PRAGMA_IN_MACRO_BODY( omp for )                                         \
      for( int k = PV_izl; k <= PV_izu; k++){                                 \
        for( int j = PV_iyl; j <= PV_iyu; j++){                               \
          for( int i = PV_ixl; i <= PV_ixu; i++){                             \
            body;                                                             \
          } /* for i */                                                       \
        } /* for j */                                                         \
      } /* for k */                                                           \
    } /* close for box */                                                     \
  } /* close parallel section */                                              \
} /* close macro block */

/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel in boxes
 *   Parallel in boxes, parallel on the Z loop (outermost loop),
 *  collapsing the Z and Y loops.
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesParallelInBoxesOnZCollapseZY(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  PRAGMA_IN_MACRO_BODY( omp parallel )                                        \
  {                                                                           \
    for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)               \
    {                                                                         \
      Box box = BoxArrayGetBox(boxes, PV_box);                                \
      /* find octree and region intersection */                               \
      int PV_ixl = pfmax(ix, box.lo[0]);                                      \
      int PV_iyl = pfmax(iy, box.lo[1]);                                      \
      int PV_izl = pfmax(iz, box.lo[2]);                                      \
      int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                           \
      int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                           \
      int PV_izu = pfmin((iz + nz - 1), box.up[2]);                           \
      PRAGMA_IN_MACRO_BODY( omp for collapse(2) )                             \
      for( int k = PV_izl; k <= PV_izu; k++){                                 \
        for( int j = PV_iyl; j <= PV_iyu; j++){                               \
          for( int i = PV_ixl; i <= PV_ixu; i++){                             \
            body;                                                             \
          } /* for i */                                                       \
        } /* for j */                                                         \
      } /* for k */                                                           \
    } /* close for box */                                                     \
  } /* close parallel section */                                              \
} /* close macro block */

/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel in boxes
 *   Parallel in boxes, parallel on the Z loop (outermost loop),
 *  collapsing the Z, Y, and X loops.
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesParallelInBoxesOnZCollapseZYX(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  PRAGMA_IN_MACRO_BODY( omp parallel )                                        \
  {                                                                           \
    for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)               \
    {                                                                         \
      Box box = BoxArrayGetBox(boxes, PV_box);                                \
      /* find octree and region intersection */                               \
      int PV_ixl = pfmax(ix, box.lo[0]);                                      \
      int PV_iyl = pfmax(iy, box.lo[1]);                                      \
      int PV_izl = pfmax(iz, box.lo[2]);                                      \
      int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                           \
      int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                           \
      int PV_izu = pfmin((iz + nz - 1), box.up[2]);                           \
      PRAGMA_IN_MACRO_BODY( omp for collapse(3) )                             \
      for( int k = PV_izl; k <= PV_izu; k++){                                 \
        for( int j = PV_iyl; j <= PV_iyu; j++){                               \
          for( int i = PV_ixl; i <= PV_ixu; i++){                             \
            body;                                                             \
          } /* for i */                                                       \
        } /* for j */                                                         \
      } /* for k */                                                           \
    } /* close for box */                                                     \
  } /* close parallel section */                                              \
} /* close macro block */

/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel in boxes
 *   Parallel in boxes, parallel on the Y loop (middleist loop)
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesParallelInBoxesOnY(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  PRAGMA_IN_MACRO_BODY( omp parallel )                                        \
  {                                                                           \
    for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)               \
    {                                                                         \
      Box box = BoxArrayGetBox(boxes, PV_box);                                \
      /* find octree and region intersection */                               \
      int PV_ixl = pfmax(ix, box.lo[0]);                                      \
      int PV_iyl = pfmax(iy, box.lo[1]);                                      \
      int PV_izl = pfmax(iz, box.lo[2]);                                      \
      int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                           \
      int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                           \
      int PV_izu = pfmin((iz + nz - 1), box.up[2]);                           \
      for( int k = PV_izl; k <= PV_izu; k++){                                 \
        PRAGMA_IN_MACRO_BODY( omp for )                                       \
        for( int j = PV_iyl; j <= PV_iyu; j++){                               \
          for( int i = PV_ixl; i <= PV_ixu; i++){                             \
            body;                                                             \
          } /* for i */                                                       \
        } /* for j */                                                         \
      } /* for k */                                                           \
    } /* close for box */                                                     \
  } /* close parallel section */                                              \
} /* close macro block */

/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel in boxes
 *   Parallel in boxes, parallel on the Y loop (middleist loop),
 *   collapsing the Y and X loops
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesParallelInBoxesOnYCollapseYX(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  PRAGMA_IN_MACRO_BODY( omp parallel )                                        \
  {                                                                           \
    for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)               \
    {                                                                         \
      Box box = BoxArrayGetBox(boxes, PV_box);                                \
      /* find octree and region intersection */                               \
      int PV_ixl = pfmax(ix, box.lo[0]);                                      \
      int PV_iyl = pfmax(iy, box.lo[1]);                                      \
      int PV_izl = pfmax(iz, box.lo[2]);                                      \
      int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                           \
      int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                           \
      int PV_izu = pfmin((iz + nz - 1), box.up[2]);                           \
      for( int k = PV_izl; k <= PV_izu; k++){                                 \
        PRAGMA_IN_MACRO_BODY( omp for collapse(2) )                           \
        for( int j = PV_iyl; j <= PV_iyu; j++){                               \
          for( int i = PV_ixl; i <= PV_ixu; i++){                             \
            body;                                                             \
          } /* for i */                                                       \
        } /* for j */                                                         \
      } /* for k */                                                           \
    } /* close for box */                                                     \
  } /* close parallel section */                                              \
} /* close macro block */

/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel in boxes
 *   Parallel in boxes, parallel on the X loop (interior-most loop)
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesParallelInBoxesOnX(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  PRAGMA_IN_MACRO_BODY( omp parallel )                                        \
  {                                                                           \
    for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)               \
    {                                                                         \
      Box box = BoxArrayGetBox(boxes, PV_box);                                \
      /* find octree and region intersection */                               \
      int PV_ixl = pfmax(ix, box.lo[0]);                                      \
      int PV_iyl = pfmax(iy, box.lo[1]);                                      \
      int PV_izl = pfmax(iz, box.lo[2]);                                      \
      int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                           \
      int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                           \
      int PV_izu = pfmin((iz + nz - 1), box.up[2]);                           \
      for( int k = PV_izl; k <= PV_izu; k++){                                 \
        for( int j = PV_iyl; j <= PV_iyu; j++){                               \
          PRAGMA_IN_MACRO_BODY( omp for )                                     \
          for( int i = PV_ixl; i <= PV_ixu; i++){                             \
            body;                                                             \
          } /* for i */                                                       \
        } /* for j */                                                         \
      } /* for k */                                                           \
    } /* close for box */                                                     \
  } /* close parallel section */                                              \
} /* close macro block */

/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel in boxes
 *   Parallel in boxes, using a tiling method to partition work
 *--------------------------------------------------------------------------*/

#ifndef GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_min
#define GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_min 16
#endif

#ifndef GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_x
#define GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_x 100
#endif

#ifndef GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_y
#define GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_y 100
#endif

#ifndef GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_z
#define GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_z 100
#endif

#define GrGeomInLoopBoxesParallelInBoxesTiled(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)                                 \
{                                                                                                                            \
  int *PV_visiting = NULL;                                                                                                   \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                                                                        \
  PRAGMA_IN_MACRO_BODY( omp parallel )                                                                                       \
  {                                                                                                                          \
    for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                                                              \
    {                                                                                                                        \
      Box box = BoxArrayGetBox(boxes, PV_box);                                                                               \
      /* find octree and region intersection */                                                                              \
      int PV_ixl = pfmax(ix, box.lo[0]);                                                                                     \
      int PV_iyl = pfmax(iy, box.lo[1]);                                                                                     \
      int PV_izl = pfmax(iz, box.lo[2]);                                                                                     \
      int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                                                                          \
      int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                                                                          \
      int PV_izu = pfmin((iz + nz - 1), box.up[2]);                                                                          \
      PRAGMA_IN_MACRO_BODY( omp for private(i,j,k) collapse(3))                                                              \
      for( int PV_tile_z = PV_izl; PV_tile_z <= PV_izu; PV_tile_z += GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_z )     \
        for( int PV_tile_y = PV_iyl; PV_tile_y <= PV_iyu; PV_tile_y += GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_y )   \
          for( int PV_tile_x = PV_ixl; PV_tile_x <= PV_ixu; PV_tile_x += GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_x ) \
          {                                                                                                                  \
            const int PV_tile_upper_x = pfmin( PV_tile_x +  GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_x - 1, PV_ixu ); \
            const int PV_tile_upper_y = pfmin( PV_tile_y +  GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_z - 1, PV_iyu ); \
            const int PV_tile_upper_z = pfmin( PV_tile_z +  GrGeomInLoopBoxesParallelInBoxesTiled_tile_size_y - 1, PV_izu ); \
            for(k = PV_tile_z; k <= PV_tile_upper_z; ++k)                                                                    \
              for(j = PV_tile_y; j <= PV_tile_upper_y; ++j)                                                                  \
                for(i = PV_tile_x; i <= PV_tile_upper_x; ++i)                                                                \
                {                                                                                                            \
                  body;                                                                                                      \
                } /* for i */                                                                                                \
          } /* close for PV_tile_x */                                                                                        \
      } /* close for PV_box */                                                                                               \
  } /* close parallel section */                                                                                             \
} /* close macro block */

/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel over and in boxes
 *   Parallel over boxes using schedule(dynamic) with default chunck size and
 *   parallel within boxes
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesParallelOverAndInBoxes(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  PRAGMA_IN_MACRO_BODY( omp parallel for private(i,j,k) schedule(dynamic))    \
  for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                 \
  {                                                                           \
    Box box = BoxArrayGetBox(boxes, PV_box);                                  \
    /* find octree and region intersection */                                 \
    int PV_ixl = pfmax(ix, box.lo[0]);                                        \
    int PV_iyl = pfmax(iy, box.lo[1]);                                        \
    int PV_izl = pfmax(iz, box.lo[2]);                                        \
    int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                             \
    int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                             \
    int PV_izu = pfmin((iz + nz - 1), box.up[2]);                             \
    PRAGMA_IN_MACRO_BODY( omp parallel for private(i,j,k) collapse(3) )       \
    for(k = PV_izl; k <= PV_izu; k++)                                         \
      for(j =PV_iyl; j <= PV_iyu; j++)                                        \
        for(i = PV_ixl; i <= PV_ixu; i++)                                     \
        {                                                                     \
          body;                                                               \
        }                                                                     \
   }                                                                          \
}

/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel over and in boxes
 *   Parallel over boxes using schedule(dynamic) with default chunck size and
 *   parallel within boxes using collapse(3)
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesParallelOverAndInCollapseBoxes(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  PRAGMA_IN_MACRO_BODY( omp parallel for private(i,j,k) schedule(dynamic))    \
  for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                 \
  {                                                                           \
    Box box = BoxArrayGetBox(boxes, PV_box);                                  \
    /* find octree and region intersection */                                 \
    int PV_ixl = pfmax(ix, box.lo[0]);                                        \
    int PV_iyl = pfmax(iy, box.lo[1]);                                        \
    int PV_izl = pfmax(iz, box.lo[2]);                                        \
    int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                             \
    int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                             \
    int PV_izu = pfmin((iz + nz - 1), box.up[2]);                             \
    PRAGMA_IN_MACRO_BODY( omp parallel for private(i,j,k) collapse(3) )       \
    for(k = PV_izl; k <= PV_izu; k++)                                         \
      for(j =PV_iyl; j <= PV_iyu; j++)                                        \
        for(i = PV_ixl; i <= PV_ixu; i++)                                     \
        {                                                                     \
          body;                                                               \
        }                                                                     \
   }                                                                          \
}

/*--------------------------------------------------------------------------
 * GrGeomSolid parallel looping macro:
 *   Macro for looping over the inside of a solid, in parallel over and in boxes
 *   Strategic parallelism grouping parallelism over small boxes
 *   and within large boxes, doing one group as a whole, then the other group
 *--------------------------------------------------------------------------*/
#ifndef GrGeomInLoopBoxesParallelSplitStrategyTwice_minimal_size_value
#define GrGeomInLoopBoxesParallelSplitStrategyTwice_minimal_size_value 10
#endif

#define GrGeomInLoopBoxesParallelSplitStrategyTwice(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)      \
{                                                                                                 \
  int *PV_visiting = NULL;                                                                        \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                                             \
  const int minimal_size = (GrGeomInLoopBoxesParallelSplitStrategyTwice_minimal_size_value); /* smallets size of 'large' boxes */ \
  int sum_of_small_box_sizes = 0;                                                                 \
  int number_of_small_boxes = 0;                                                                  \
  /* Do large boxes; collect small box size metrics */                                            \
  {                                                                                               \
    for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                                   \
    {                                                                                             \
      Box box = BoxArrayGetBox(boxes, PV_box);                                                    \
      /* find octree and region intersection */                                                   \
      int PV_ixl = pfmax(ix, box.lo[0]);                                                          \
      int PV_iyl = pfmax(iy, box.lo[1]);                                                          \
      int PV_izl = pfmax(iz, box.lo[2]);                                                          \
      int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                                               \
      int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                                               \
      int PV_izu = pfmin((iz + nz - 1), box.up[2]);                                               \
      int size = (PV_ixu-PV_ixl+1)*(PV_iyu-PV_iyl+1)*(PV_izu-PV_izl+1);                           \
      if( size >= minimal_size ) {                                                                \
        PRAGMA_IN_MACRO_BODY( omp parallel for private(i,j,k) schedule(static) )                  \
        for(k = PV_izl; k <= PV_izu; k++)                                                         \
          for(j = PV_iyl; j <= PV_iyu; j++)                                                       \
            for(i = PV_ixl; i <= PV_ixu; i++)                                                     \
            {                                                                                     \
              body;                                                                               \
            }                                                                                     \
      } else if( size > 0 ){                                                                      \
        sum_of_small_box_sizes += size;                                                           \
        number_of_small_boxes += 1;                                                               \
      } else {                                                                                    \
        /* In case there are invalid boxes s.t. the stop point is behind the start point */       \
        /* ideally, this should not happen */                                                     \
      }                                                                                           \
    }                                                                                             \
  }                                                                                               \
  /* Do small boxes */                                                                            \
  if( number_of_small_boxes > 0 ){                                                                \
    /* create an idealized average small-box size, use that to calculate an apprximation */       \
    /* of how many of those boxes there are. ideally this would be the chunk size; */             \
    /* however, we have to contend with the number of big boxes. Add those to the count */        \
    /* Assume all are evenly distributed */                                                       \
    int mean_small_box_size = sum_of_small_box_sizes / number_of_small_boxes;                     \
    int number_of_mean_sized_boxes = sum_of_small_box_sizes / mean_small_box_size;                \
    int number_of_large_boxes = BoxArraySize(boxes) - number_of_small_boxes;                      \
    int ideal_chunk_size = (sum_of_small_box_sizes / mean_small_box_size) / omp_get_num_threads();\
    {                                                                                             \
      PRAGMA_IN_MACRO_BODY( omp parallel for private(i,j,k) schedule(dynamic, ideal_chunk_size))  \
      for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                                 \
      {                                                                                           \
        Box box = BoxArrayGetBox(boxes, PV_box);                                                  \
        /* find octree and region intersection */                                                 \
        int PV_ixl = pfmax(ix, box.lo[0]);                                                        \
        int PV_iyl = pfmax(iy, box.lo[1]);                                                        \
        int PV_izl = pfmax(iz, box.lo[2]);                                                        \
        int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                                             \
        int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                                             \
        int PV_izu = pfmin((iz + nz - 1), box.up[2]);                                             \
        int size = (PV_ixu-PV_ixl+1)*(PV_iyu-PV_iyl+1)*(PV_izu-PV_izl+1);                         \
        if( size < minimal_size ) {                                                               \
          for(k = PV_izl; k <= PV_izu; k++)                                                       \
            for(j = PV_iyl; j <= PV_iyu; j++)                                                     \
              for(i = PV_ixl; i <= PV_ixu; i++)                                                   \
              {                                                                                   \
                body;                                                                             \
              }                                                                                   \
        }                                                                                         \
      }                                                                                           \
    }                                                                                             \
  }                                                                                               \
}

/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the entire domain
 *   Serial in every way
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesTotalDomain(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body) \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  /* for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++) */           \
  if( BoxArraySize(boxes) > 1 ){                                              \
    printf("More than one box (%d)\n", BoxArraySize(boxes));                  \
  }                                                                           \
  int PV_box = 0;                                                             \
  {                                                                           \
    Box box = BoxArrayGetBox(boxes, PV_box);                                  \
    /* find octree and region intersection */                                 \
    int PV_ixl = pfmax(ix, box.lo[0]);                                        \
    int PV_iyl = pfmax(iy, box.lo[1]);                                        \
    int PV_izl = pfmax(iz, box.lo[2]);                                        \
    int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                             \
    int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                             \
    int PV_izu = pfmin((iz + nz - 1), box.up[2]);                             \
    for(k = PV_izl; k <= PV_izu; k++)                                         \
      for(j = PV_iyl; j <= PV_iyu; j++)                                       \
        for(i = PV_ixl; i <= PV_ixu; i++)                                     \
        {                                                                     \
          body;                                                               \
        }                                                                     \
   }                                                                          \
}



/*
 Note: this parameter is used in all GrGeomInLoopBoxesTotalDomainTiled macros
*/
 #ifndef GrGeomInLoopBoxesTotalDomainTiled_tile_size
 #define GrGeomInLoopBoxesTotalDomainTiled_tile_size (10)
 #endif

 /*--------------------------------------------------------------------------
  * GrGeomSolid looping macro:
  *   Macro for looping over the entire domain in a tiled manner
  *   Serial
  *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesTotalDomainTiled(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body) \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  /* for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++) */           \
  if( BoxArraySize(boxes) > 1 ){                                              \
    printf("More than one box (%d)\n", BoxArraySize(boxes));                  \
  }                                                                           \
  int PV_box = 0;                                                             \
  {                                                                           \
    Box box = BoxArrayGetBox(boxes, PV_box);                                  \
    /* find octree and region intersection */                                 \
    int PV_ixl = pfmax(ix, box.lo[0]);                                        \
    int PV_iyl = pfmax(iy, box.lo[1]);                                        \
    int PV_izl = pfmax(iz, box.lo[2]);                                        \
    int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                             \
    int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                             \
    int PV_izu = pfmin((iz + nz - 1), box.up[2]);                             \
    for( int PV_tile_z = PV_izl; PV_tile_z <= PV_izu; PV_tile_z += GrGeomInLoopBoxesTotalDomainTiled_tile_size )     \
      for( int PV_tile_y = PV_iyl; PV_tile_y <= PV_iyu; PV_tile_y += GrGeomInLoopBoxesTotalDomainTiled_tile_size )   \
        for( int PV_tile_x = PV_ixl; PV_tile_x <= PV_ixu; PV_tile_x += GrGeomInLoopBoxesTotalDomainTiled_tile_size ) \
        {                                                                               \
          const int PV_tile_upper_x = pfmin( PV_tile_x +  GrGeomInLoopBoxesTotalDomainTiled_tile_size - 1, PV_ixu ); \
          const int PV_tile_upper_y = pfmin( PV_tile_y +  GrGeomInLoopBoxesTotalDomainTiled_tile_size - 1, PV_iyu ); \
          const int PV_tile_upper_z = pfmin( PV_tile_z +  GrGeomInLoopBoxesTotalDomainTiled_tile_size - 1, PV_izu ); \
          for(k = PV_tile_z; k <= PV_tile_upper_z; ++k)                                             \
            for(j = PV_tile_y; j <= PV_tile_upper_y; ++j)                                           \
              for(i = PV_tile_x; i <= PV_tile_upper_x; ++i)                                         \
              {                                                                         \
                body;                                                                   \
              }                                                                         \
        }                                                                               \
    }                                                                                   \
}

 /*--------------------------------------------------------------------------
  * GrGeomSolid looping macro:
  *   Macro for looping over the entire domain in a tiled manner
  *   Parallel over tiles
  *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesTotalDomainTiledParallelOverTiles(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body) \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  /* for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++) */           \
  if( BoxArraySize(boxes) > 1 ){                                              \
    printf("More than one box (%d)\n", BoxArraySize(boxes));                  \
  }                                                                           \
  int PV_box = 0;                                                             \
  {                                                                           \
    Box box = BoxArrayGetBox(boxes, PV_box);                                  \
    /* find octree and region intersection */                                 \
    int PV_ixl = pfmax(ix, box.lo[0]);                                        \
    int PV_iyl = pfmax(iy, box.lo[1]);                                        \
    int PV_izl = pfmax(iz, box.lo[2]);                                        \
    int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                             \
    int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                             \
    int PV_izu = pfmin((iz + nz - 1), box.up[2]);                             \
    PRAGMA_IN_MACRO_BODY( omp parallel for collapse(3) schedule(static) private(i, j, k) )               \
    for( int PV_tile_z = PV_izl; PV_tile_z <= PV_izu; PV_tile_z += GrGeomInLoopBoxesTotalDomainTiled_tile_size )     \
      for( int PV_tile_y = PV_iyl; PV_tile_y <= PV_iyu; PV_tile_y += GrGeomInLoopBoxesTotalDomainTiled_tile_size )   \
        for( int PV_tile_x = PV_ixl; PV_tile_x <= PV_ixu; PV_tile_x += GrGeomInLoopBoxesTotalDomainTiled_tile_size ) \
        {                                                                               \
          const int PV_tile_upper_x = pfmin( PV_tile_x +  GrGeomInLoopBoxesTotalDomainTiled_tile_size - 1, PV_ixu ); \
          const int PV_tile_upper_y = pfmin( PV_tile_y +  GrGeomInLoopBoxesTotalDomainTiled_tile_size - 1, PV_iyu ); \
          const int PV_tile_upper_z = pfmin( PV_tile_z +  GrGeomInLoopBoxesTotalDomainTiled_tile_size - 1, PV_izu ); \
          for(k = PV_tile_z; k <= PV_tile_upper_z; ++k)                                             \
            for(j = PV_tile_y; j <= PV_tile_upper_y; ++j)                                           \
              for(i = PV_tile_x; i <= PV_tile_upper_x; ++i)                                         \
              {                                                                         \
                body;                                                                   \
              }                                                                         \
        }                                                                               \
    }                                                                                   \
}

/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the entire domain in a tiled manner
 *   Parallel in tiles
 *--------------------------------------------------------------------------*/

#define GrGeomInLoopBoxesTotalDomainTiledParallelInTiles(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body)   \
{                                                                             \
  int *PV_visiting = NULL;                                                    \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                         \
  /* for(int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++) */           \
  if( BoxArraySize(boxes) > 1 ){                                              \
    printf("More than one box (%d)\n", BoxArraySize(boxes));                  \
  }                                                                           \
  int PV_box = 0;                                                             \
  {                                                                           \
    Box box = BoxArrayGetBox(boxes, PV_box);                                  \
    /* find octree and region intersection */                                 \
    int PV_ixl = pfmax(ix, box.lo[0]);                                        \
    int PV_iyl = pfmax(iy, box.lo[1]);                                        \
    int PV_izl = pfmax(iz, box.lo[2]);                                        \
    int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                             \
    int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                             \
    int PV_izu = pfmin((iz + nz - 1), box.up[2]);                             \
    for( int PV_tile_z = PV_izl; PV_tile_z <= PV_izu; PV_tile_z += GrGeomInLoopBoxesTotalDomainTiled_tile_size )     \
      for( int PV_tile_y = PV_iyl; PV_tile_y <= PV_iyu; PV_tile_y += GrGeomInLoopBoxesTotalDomainTiled_tile_size )   \
        for( int PV_tile_x = PV_ixl; PV_tile_x <= PV_ixu; PV_tile_x += GrGeomInLoopBoxesTotalDomainTiled_tile_size ) \
        {                                                                               \
          const int PV_tile_upper_x = pfmin( PV_tile_x +  GrGeomInLoopBoxesTotalDomainTiled_tile_size - 1, PV_ixu ); \
          const int PV_tile_upper_y = pfmin( PV_tile_y +  GrGeomInLoopBoxesTotalDomainTiled_tile_size - 1, PV_iyu ); \
          const int PV_tile_upper_z = pfmin( PV_tile_z +  GrGeomInLoopBoxesTotalDomainTiled_tile_size - 1, PV_izu ); \
          PRAGMA_IN_MACRO_BODY( omp parallel for collapse(3) schedule(static) private(i, j, k) )               \
          for(k = PV_tile_z; k <= PV_tile_upper_z; ++k)                                             \
            for(j = PV_tile_y; j <= PV_tile_upper_y; ++j)                                           \
              for(i = PV_tile_x; i <= PV_tile_upper_x; ++i)                                         \
              {                                                                         \
                body;                                                                   \
              }                                                                         \
        }                                                                               \
    }                                                                                   \
}

#define GrGeomInLoop(i, j, k, grgeom, r, ix, iy, iz, nx, ny, nz, body)        \
  {                                                                           \
   if(r == 0 && GrGeomSolidInteriorBoxes(grgeom))                             \
   {                                                                          \
     GrGeomInLoopBoxes(i, j, k, grgeom,                                       \
		       ix, iy, iz, nx, ny, nz, body);                                     \
   }                                                                          \
   else                                                                       \
   {                                                                          \
     GrGeomOctree  *PV_node;                                                  \
     double PV_ref = pow(2.0, r);                                             \
                                                                              \
     i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                           \
     j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                           \
     k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                           \
     GrGeomOctreeInteriorNodeLoop(i, j, k, PV_node,                           \
				  GrGeomSolidData(grgeom),                                            \
				  GrGeomSolidOctreeBGLevel(grgeom) + r,                               \
				  ix, iy, iz, nx, ny, nz,                                             \
				  TRUE,                                                               \
				  body);                                                              \
   }                                                                          \
  }


// Change for different versions
#ifndef GrGeomInLoopBoxesParallel
  #define GrGeomInLoopBoxesParallel GrGeomInLoopBoxes
#endif

#define GrGeomInLoopParallel(i, j, k, grgeom, r, ix, iy, iz, nx, ny, nz, body)	\
  {                                                                           \
   if(r == 0 && GrGeomSolidInteriorBoxes(grgeom))                             \
   {                                                                          \
     GrGeomInLoopBoxesParallel(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body); \
   }                                                                          \
   else                                                                       \
   {                                                                          \
     GrGeomOctree  *PV_node;                                                  \
     double PV_ref = pow(2.0, r);                                             \
                                                                              \
     i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                           \
     j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                           \
     k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                           \
     GrGeomOctreeInteriorNodeLoop(i, j, k, PV_node,                           \
       GrGeomSolidData(grgeom),                                               \
       GrGeomSolidOctreeBGLevel(grgeom) + r,                                  \
       ix, iy, iz, nx, ny, nz,                                                \
       TRUE,                                                                  \
       body                                                                   \
     );                                                                       \
   }                                                                          \
}


/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the inside of a solid with non-unitary strides.
 *--------------------------------------------------------------------------*/

/**
 *  Interior version of this would improve speed; but this loop is not
 * currently used.
 */
#define GrGeomInLoop2(i, j, k, grgeom,                             \
                      r, ix, iy, iz, nx, ny, nz, sx, sy, sz, body) \
  {                                                                \
    GrGeomOctree  *PV_node;                                        \
    double PV_ref = pow(2.0, r);                                   \
                                                                   \
                                                                   \
    i = GrGeomSolidOctreeIX(grgeom) * PV_ref;                      \
    j = GrGeomSolidOctreeIY(grgeom) * PV_ref;                      \
    k = GrGeomSolidOctreeIZ(grgeom) * PV_ref;                      \
    GrGeomOctreeNodeLoop2(i, j, k, PV_node,                        \
                          GrGeomSolidData(grgeom),                 \
                          GrGeomSolidOctreeBGLevel(grgeom) + r,    \
                          ix, iy, iz, nx, ny, nz, sx, sy, sz,      \
                          (GrGeomOctreeNodeIsInside(PV_node) ||    \
                           GrGeomOctreeNodeIsFull(PV_node)),       \
                          body);                                   \
  }

/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the outside of a solid.
 *--------------------------------------------------------------------------*/

#define GrGeomOutLoop(i, j, k, grgeom,                                 \
                      r, ix, iy, iz, nx, ny, nz, body)                 \
  {                                                                    \
    GrGeomOctree  *PV_node;                                            \
    double PV_ref = pow(2.0, r);                                       \
                                                                       \
                                                                       \
    i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                     \
    j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                     \
    k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                     \
    GrGeomOctreeExteriorNodeLoop(i, j, k, PV_node,                     \
                                 GrGeomSolidData(grgeom),              \
                                 GrGeomSolidOctreeBGLevel(grgeom) + r, \
                                 ix, iy, iz, nx, ny, nz,               \
                                 TRUE,                                 \
                                 body);                                \
  }

/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the outside of a solid with non-unitary strides.
 *--------------------------------------------------------------------------*/

#define GrGeomOutLoop2(i, j, k, grgeom,                             \
                       r, ix, iy, iz, nx, ny, nz, sx, sy, sz, body) \
  {                                                                 \
    GrGeomOctree  *PV_node;                                         \
    double PV_ref = pow(2.0, r);                                    \
                                                                    \
                                                                    \
    i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                  \
    j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                  \
    k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                  \
    GrGeomOctreeNodeLoop2(i, j, k, PV_node,                         \
                          GrGeomSolidData(grgeom),                  \
                          GrGeomSolidOctreeBGLevel(grgeom) + r,     \
                          ix, iy, iz, nx, ny, nz, sx, sy, sz,       \
                          (GrGeomOctreeNodeIsOutside(PV_node) ||    \
                           GrGeomOctreeNodeIsEmpty(PV_node)),       \
                          body);                                    \
  }

/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the faces of a solid surface.
 *--------------------------------------------------------------------------*/
#if 0
#define GrGeomSurfLoop(i, j, k, fdir, grgeom,                  \
                       r, ix, iy, iz, nx, ny, nz, body)        \
  {                                                            \
    GrGeomOctree  *PV_node;                                    \
    double PV_ref = pow(2.0, r);                               \
                                                               \
    i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;             \
    j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;             \
    k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;             \
    GrGeomOctreeFaceLoop(i, j, k, fdir, PV_node,               \
                         GrGeomSolidData(grgeom),              \
                         GrGeomSolidOctreeBGLevel(grgeom) + r, \
                         ix, iy, iz, nx, ny, nz, body);        \
  }

#else

//  \todo SGS 12/3/2008 can optimize fdir by using 1 assignment to static.  Should
// elimiate 2 assignment statements and switch and replace with table:
// fdir = FDIR[PV_f] type of thing.
//

#define GrGeomSurfLoopBoxes(i, j, k, fdir, grgeom, ix, iy, iz, nx, ny, nz, body) \
  {                                                                              \
    int PV_fdir[3];                                                              \
                                                                                 \
    fdir = PV_fdir;                                                              \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;                          \
    int *PV_visiting = NULL;                                                     \
    for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)                      \
    {                                                                            \
      switch (PV_f)                                                              \
      {                                                                          \
        case GrGeomOctreeFaceL:                                                  \
          fdir[0] = -1; fdir[1] = 0; fdir[2] = 0;                                \
          break;                                                                 \
        case GrGeomOctreeFaceR:                                                  \
          fdir[0] = 1; fdir[1] = 0; fdir[2] = 0;                                 \
          break;                                                                 \
        case GrGeomOctreeFaceD:                                                  \
          fdir[0] = 0; fdir[1] = -1; fdir[2] = 0;                                \
          break;                                                                 \
        case GrGeomOctreeFaceU:                                                  \
          fdir[0] = 0; fdir[1] = 1; fdir[2] = 0;                                 \
          break;                                                                 \
        case GrGeomOctreeFaceB:                                                  \
          fdir[0] = 0; fdir[1] = 0; fdir[2] = -1;                                \
          break;                                                                 \
        case GrGeomOctreeFaceF:                                                  \
          fdir[0] = 0; fdir[1] = 0; fdir[2] = 1;                                 \
          break;                                                                 \
        default:                                                                 \
          fdir[0] = -9999; fdir[1] = -9999; fdir[2] = -99999;                    \
          break;                                                                 \
      }                                                                          \
                                                                                 \
      BoxArray* boxes = GrGeomSolidSurfaceBoxes(grgeom, PV_f);                   \
      for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)               \
      {                                                                          \
        Box box = BoxArrayGetBox(boxes, PV_box);                                 \
        /* find octree and region intersection */                                \
        PV_ixl = pfmax(ix, box.lo[0]);                                           \
        PV_iyl = pfmax(iy, box.lo[1]);                                           \
        PV_izl = pfmax(iz, box.lo[2]);                                           \
        PV_ixu = pfmin((ix + nx - 1), box.up[0]);                                \
        PV_iyu = pfmin((iy + ny - 1), box.up[1]);                                \
        PV_izu = pfmin((iz + nz - 1), box.up[2]);                                \
                                                                                 \
        for (k = PV_izl; k <= PV_izu; k++)                                       \
          for (j = PV_iyl; j <= PV_iyu; j++)                                     \
            for (i = PV_ixl; i <= PV_ixu; i++)                                   \
            {                                                                    \
              body;                                                              \
            }                                                                    \
      }                                                                          \
    }                                                                            \
  }


#define GrGeomSurfLoop(i, j, k, fdir, grgeom,                                \
                       r, ix, iy, iz, nx, ny, nz, body)                      \
  {                                                                          \
    if (r == 0 && GrGeomSolidSurfaceBoxes(grgeom, GrGeomOctreeNumFaces - 1)) \
    {                                                                        \
      GrGeomSurfLoopBoxes(i, j, k, fdir, grgeom,                             \
                          ix, iy, iz, nx, ny, nz, body);                     \
    }                                                                        \
    else                                                                     \
    {                                                                        \
      GrGeomOctree  *PV_node;                                                \
      double PV_ref = pow(2.0, r);                                           \
                                                                             \
      i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                         \
      j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                         \
      k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                         \
      GrGeomOctreeFaceLoop(i, j, k, fdir, PV_node,                           \
                           GrGeomSolidData(grgeom),                          \
                           GrGeomSolidOctreeBGLevel(grgeom) + r,             \
                           ix, iy, iz, nx, ny, nz, body);                    \
    }                                                                        \
  }

#endif



/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the faces of a solid patch.
 *--------------------------------------------------------------------------*/

#if 1

#define GrGeomPatchLoopBoxes(i, j, k, fdir, grgeom, patch_num, ix, iy, iz, nx, ny, nz, body) \
  {                                                                                          \
    int PV_fdir[3];                                                                          \
                                                                                             \
    fdir = PV_fdir;                                                                          \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;                                      \
    int *PV_visiting = NULL;                                                                 \
    for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)                                  \
    {                                                                                        \
      switch (PV_f)                                                                          \
      {                                                                                      \
        case GrGeomOctreeFaceL:                                                              \
          fdir[0] = -1; fdir[1] = 0; fdir[2] = 0;                                            \
          break;                                                                             \
        case GrGeomOctreeFaceR:                                                              \
          fdir[0] = 1; fdir[1] = 0; fdir[2] = 0;                                             \
          break;                                                                             \
        case GrGeomOctreeFaceD:                                                              \
          fdir[0] = 0; fdir[1] = -1; fdir[2] = 0;                                            \
          break;                                                                             \
        case GrGeomOctreeFaceU:                                                              \
          fdir[0] = 0; fdir[1] = 1; fdir[2] = 0;                                             \
          break;                                                                             \
        case GrGeomOctreeFaceB:                                                              \
          fdir[0] = 0; fdir[1] = 0; fdir[2] = -1;                                            \
          break;                                                                             \
        case GrGeomOctreeFaceF:                                                              \
          fdir[0] = 0; fdir[1] = 0; fdir[2] = 1;                                             \
          break;                                                                             \
        default:                                                                             \
          fdir[0] = -9999; fdir[1] = -9999; fdir[2] = -99999;                                \
          break;                                                                             \
      }                                                                                      \
                                                                                             \
      BoxArray* boxes = GrGeomSolidPatchBoxes(grgeom, patch_num, PV_f);                      \
      for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                           \
      {                                                                                      \
        Box box = BoxArrayGetBox(boxes, PV_box);                                             \
        /* find octree and region intersection */                                            \
        PV_ixl = pfmax(ix, box.lo[0]);                                                       \
        PV_iyl = pfmax(iy, box.lo[1]);                                                       \
        PV_izl = pfmax(iz, box.lo[2]);                                                       \
        PV_ixu = pfmin((ix + nx - 1), box.up[0]);                                            \
        PV_iyu = pfmin((iy + ny - 1), box.up[1]);                                            \
        PV_izu = pfmin((iz + nz - 1), box.up[2]);                                            \
                                                                                             \
        for (k = PV_izl; k <= PV_izu; k++)                                                   \
          for (j = PV_iyl; j <= PV_iyu; j++)                                                 \
            for (i = PV_ixl; i <= PV_ixu; i++)                                               \
            {                                                                                \
              body;                                                                          \
            }                                                                                \
      }                                                                                      \
    }                                                                                        \
  }

#define GrGeomPatchLoop(i, j, k, fdir, grgeom, patch_num,                             \
                        r, ix, iy, iz, nx, ny, nz, body)                              \
  {                                                                                   \
    if (r == 0 && GrGeomSolidPatchBoxes(grgeom, patch_num, GrGeomOctreeNumFaces - 1)) \
    {                                                                                 \
      GrGeomPatchLoopBoxes(i, j, k, fdir, grgeom, patch_num,                          \
                           ix, iy, iz, nx, ny, nz, body);                             \
    }                                                                                 \
    else                                                                              \
    {                                                                                 \
      GrGeomOctree  *PV_node;                                                         \
      double PV_ref = pow(2.0, r);                                                    \
                                                                                      \
                                                                                      \
      i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                                  \
      j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                                  \
      k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                                  \
      GrGeomOctreeFaceLoop(i, j, k, fdir, PV_node,                                    \
                           GrGeomSolidPatch(grgeom, patch_num),                       \
                           GrGeomSolidOctreeBGLevel(grgeom) + r,                      \
                           ix, iy, iz, nx, ny, nz, body);                             \
    }                                                                                 \
  }

#else

#define GrGeomPatchLoop(i, j, k, fdir, grgeom, patch_num,      \
                        r, ix, iy, iz, nx, ny, nz, body)       \
  {                                                            \
    GrGeomOctree  *PV_node;                                    \
    double PV_ref = pow(2.0, r);                               \
                                                               \
                                                               \
    i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;             \
    j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;             \
    k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;             \
    GrGeomOctreeFaceLoop(i, j, k, fdir, PV_node,               \
                         GrGeomSolidPatch(grgeom, patch_num),  \
                         GrGeomSolidOctreeBGLevel(grgeom) + r, \
                         ix, iy, iz, nx, ny, nz, body);        \
  }

#endif


/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 * Macro for looping over the inside of a solid as a set of boxes.
 * This will pick both active and inactive cells along the boundary but
 * will avoid regions that are totally inactive.
 *
 * This may be used in place of the GrGeomInLoop macro if computing in
 * the inactive region is OK but not desired.  The advantage over the
 * GrGeomInLoop is this may be more computationally efficient as it is
 * looping over box (patch/block) of data rather than potentially
 * looping over individual elements.
 *
 * int i,j,k                     the starting index values for each box.
 * int num_i, num_j, num_k       the number of points in each box.
 * grgeom                        GrGeomSolid to loop over.
 * box_size_power                Smallest size of box to loop over as a power of 2.
 *                               The boxes will be 2^box_size_power cubed.
 *                               Boxes may be larger than this for interior regions.
 *                               Boxes may be smaller than this near boundaries.
 *                               Power of 2 restriction is imposed by the octree
 *                               representation.
 *--------------------------------------------------------------------------*/

// Note that IsInside is here to make sure everything is
// included.   If IsInside is actually true then that means single
// cells are being looped over which would be really bad for
// performance reasons.

#define GrGeomInBoxLoop(                                          \
                        i, j, k,                                  \
                        num_i, num_j, num_k,                      \
                        grgeom, box_size_power,                   \
                        ix, iy, iz, nx, ny, nz,                   \
                        body)                                     \
  {                                                               \
    GrGeomOctree  *PV_node;                                       \
    int PV_level_of_interest;                                     \
    PV_level_of_interest = GrGeomSolidOctreeBGLevel(grgeom) -     \
                           box_size_power - 1;                    \
    PV_level_of_interest = pfmax(0, PV_level_of_interest);        \
                                                                  \
    i = GrGeomSolidOctreeIX(grgeom);                              \
    j = GrGeomSolidOctreeIY(grgeom);                              \
    k = GrGeomSolidOctreeIZ(grgeom);                              \
                                                                  \
    GrGeomOctreeNodeBoxLoop(i, j, k,                              \
                            num_i, num_j, num_k,                  \
                            PV_node,                              \
                            GrGeomSolidData(grgeom),              \
                            GrGeomSolidOctreeBGLevel(grgeom),     \
                            PV_level_of_interest,                 \
                            ix, iy, iz, nx, ny, nz,               \
                            (GrGeomOctreeHasChildren(PV_node) ||  \
                             GrGeomOctreeNodeIsInside(PV_node) || \
                             GrGeomOctreeNodeIsFull(PV_node)),    \
    {                                                             \
      body;                                                       \
    });                                                           \
  }



#endif
