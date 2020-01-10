#ifndef _PF_OMP_GRGEOM_H
#define _PF_OMP_GRGEOM_H

/*------------------------------------------------------------------------
 * GrGeomInLoop Redefinitions
 *------------------------------------------------------------------------*/

//#undef _GrGeomInLoop
//#define _GrGeomInLoop(pragma, ...)  pragma ## _GrGeomInLoop( __VA_ARGS__ )

/*
  It's pointless to use OpenMP if we don't have boxloop parallelism
  Until (if ever) the Octree is parallelized, just abort execution when
  compiled with OMP and not using clustering.
  This could probably be made into a check much earlier in the program, eliminating
  all of the resulting if checks.
*/
#undef _GrGeomInLoop
#define _GrGeomInLoop(pragma, locals,                                   \
                     i, j, k, grgeom, r,                                \
                     ix, iy, iz, nx, ny, nz,                            \
                     body)                                              \
  {                                                                     \
    if (r != 0 || !GrGeomSolidInteriorBoxes(grgeom))                    \
    {                                                                   \
      if(!amps_Rank(amps_CommWorld))                                    \
      {                                                                 \
        amps_Printf("Use of OpenMP requires box clustering! Aborting from:\n%s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                        \
      }                                                                 \
    }                                                                   \
    pragma ## _GrGeomInLoop( locals, i, j, k,                           \
                             grgeom, ix, iy, iz,                        \
                             nx, ny, nz,                                \
                             body );                                    \
  }

#define NewParallel_GrGeomInLoop(locals, i, j, k,                     \
                                 grgeom, ix, iy, iz,                  \
                                 nx, ny, nz, body)                    \
  {                                                                   \
    PRAGMA(omp parallel)                                              \
    {                                                                 \
      int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;             \
      int *PV_visiting = NULL;                                        \
      BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);             \
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
        PRAGMA(omp for collapse(3) private(i, j, k locals))           \
          for (k = PV_izl; k <= PV_izu; k++)                          \
          {                                                           \
            for (j = PV_iyl; j <= PV_iyu; j++)                        \
            {                                                         \
              for (i = PV_ixl; i <= PV_ixu; i++)                      \
              {                                                       \
                body;                                                 \
              }                                                       \
            }                                                         \
          }                                                           \
      }                                                               \
    }                                                                 \
  }

#define InParallel_GrGeomInLoop(locals, i, j, k,                      \
                                grgeom, ix, iy, iz,                   \
                                nx, ny, nz, body)                     \
  {                                                                   \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;               \
    int *PV_visiting = NULL;                                          \
    BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);               \
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
      PRAGMA(omp for collapse(3) private(i, j, k locals))             \
        for (k = PV_izl; k <= PV_izu; k++)                            \
        {                                                             \
          for (j = PV_iyl; j <= PV_iyu; j++)                          \
          {                                                           \
            for (i = PV_ixl; i <= PV_ixu; i++)                        \
            {                                                         \
              body;                                                   \
            }                                                         \
          }                                                           \
        }                                                             \
    }                                                                 \
  }

#define NoWait_GrGeomInLoop(locals, i, j, k,                          \
                            grgeom, ix, iy, iz,                       \
                            nx, ny, nz, body)                         \
  {                                                                   \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;               \
    int *PV_visiting = NULL;                                          \
    BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);               \
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
      PRAGMA(omp for nowait collapse(3) private(i, j, k locals))      \
        for (k = PV_izl; k <= PV_izu; k++)                            \
        {                                                             \
          for (j = PV_iyl; j <= PV_iyu; j++)                          \
          {                                                           \
            for (i = PV_ixl; i <= PV_ixu; i++)                        \
            {                                                         \
              body;                                                   \
            }                                                         \
          }                                                           \
        }                                                             \
    }                                                                 \
  }

#if 0
#undef _GrGeomInLoop
#define _GrGeomInLoop(locals, i, j, k, grgeom, r, ix, iy, iz, nx, ny, nz, body) \
  {                                                                     \
    if (r == 0 && GrGeomSolidInteriorBoxes(grgeom))                     \
    {                                                                   \
      _GrGeomInLoopBoxes(locals,                                        \
                         i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body); \
    }                                                                   \
    else                                                                \
    {                                                                   \
      GrGeomOctree  *PV_node;                                           \
      double PV_ref = pow(2.0, r);                                      \
                                                                        \
      i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                    \
      j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                    \
      k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                    \
      GrGeomOctreeInteriorNodeLoop(i, j, k, PV_node,                    \
                                   GrGeomSolidData(grgeom),             \
                                   GrGeomSolidOctreeBGLevel(grgeom) + r, \
                                   ix, iy, iz, nx, ny, nz,              \
                                   TRUE,                                \
                                   body);                               \
    }                                                                   \
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

#define __GrGeomInLoop(locals, i, j, k, grgeom, r, ix, iy, iz, nx, ny, nz, body) \
  {                                                                     \
    if (r == 0 && GrGeomSolidInteriorBoxes(grgeom))                     \
    {                                                                   \
      __GrGeomInLoopBoxes(locals,                                       \
                          i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body); \
    }                                                                   \
    else                                                                \
    {                                                                   \
      GrGeomOctree  *PV_node;                                           \
      double PV_ref = pow(2.0, r);                                      \
                                                                        \
      i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                    \
      j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                    \
      k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                    \
      GrGeomOctreeInteriorNodeLoop(i, j, k, PV_node,                    \
                                   GrGeomSolidData(grgeom),             \
                                   GrGeomSolidOctreeBGLevel(grgeom) + r, \
                                   ix, iy, iz, nx, ny, nz,              \
                                   TRUE,                                \
                                   body);                               \
    }                                                                   \
  }

#define __GrGeomInLoopBoxes(locals, i, j, k,                      \
                            grgeom, ix, iy, iz,                   \
                            nx, ny, nz, body)                     \
  {                                                               \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;           \
    int *PV_visiting = NULL;                                      \
    BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);           \
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
  }


#define ___GrGeomInLoop(locals, i, j, k, grgeom, r, ix, iy, iz, nx, ny, nz, body) \
  {                                                                     \
    if (r == 0 && GrGeomSolidInteriorBoxes(grgeom))                     \
    {                                                                   \
      ___GrGeomInLoopBoxes(locals,                                      \
                          i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body); \
    }                                                                   \
    else                                                                \
    {                                                                   \
      GrGeomOctree  *PV_node;                                           \
      double PV_ref = pow(2.0, r);                                      \
                                                                        \
      i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                    \
      j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                    \
      k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                    \
      GrGeomOctreeInteriorNodeLoop(i, j, k, PV_node,                    \
                                   GrGeomSolidData(grgeom),             \
                                   GrGeomSolidOctreeBGLevel(grgeom) + r, \
                                   ix, iy, iz, nx, ny, nz,              \
                                   TRUE,                                \
                                   body);                               \
    }                                                                   \
  }

#define ___GrGeomInLoopBoxes(locals, i, j, k,                     \
                             grgeom, ix, iy, iz,                  \
                             nx, ny, nz, body)                    \
  {                                                               \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;           \
    int *PV_visiting = NULL;                                      \
    BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);           \
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
      PRAGMA(omp for nowait collapse(3) private(i, j, k locals))  \
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
  }

#endif // Old OMP macros

#endif // _PF_OMP_GRGEOM_H
