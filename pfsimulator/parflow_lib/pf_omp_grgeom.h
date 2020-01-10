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
      }                                                                 \
      exit(1);                                                          \
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

#undef _GrGeomSurfLoop
#define _GrGeomSurfLoop(pragma, locals, i, j, k, fdir, grgeom,          \
                        r, ix, iy, iz, nx, ny, nz, body)                \
  {                                                                     \
    if (r != 0 || !GrGeomSolidSurfaceBoxes(grgeom, GrGeomOctreeNumFaces - 1)) \
    {                                                                   \
      if(!amps_Rank(amps_CommWorld))                                    \
      {                                                                 \
        amps_Printf("Use of OpenMP requires box clustering! Aborting from:\n%s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                        \
      }                                                                 \
    }                                                                   \
    pragma ## _GrGeomSurfLoopBoxes(locals, i, j, k, fdir, grgeom,       \
                                   ix, iy, iz, nx, ny, nz, body);       \
  }

#define NewParallel_GrGeomSurfLoopBoxes(locals,                         \
                                        i, j, k,                        \
                                        fdir, grgeom,                   \
                                        ix, iy, iz,                     \
                                        nx, ny, nz,                     \
                                        body)                           \
  {                                                                     \
    PRAGMA(omp parallel private(fdir))                                  \
    {                                                                   \
                                                                        \
      int PV_fdir[3];                                                   \
      int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;               \
      int *PV_visiting = NULL;                                          \
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
        BoxArray* boxes = GrGeomSolidSurfaceBoxes(grgeom, PV_f);        \
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
    }                                                                   \
  }

#define InParallel_GrGeomSurfLoopBoxes(locals,                    \
                                       i, j, k,                   \
                                       fdir, grgeom,              \
                                       ix, iy, iz,                \
                                       nx, ny, nz,                \
                                       body)                      \
  {                                                               \
    int PV_fdir[3];                                               \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;           \
    int *PV_visiting = NULL;                                      \
    fdir = PV_fdir;                                               \
    for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)       \
    {                                                             \
      switch (PV_f)                                               \
      {                                                           \
        case GrGeomOctreeFaceL:                                   \
          fdir[0] = -1; fdir[1] = 0; fdir[2] = 0;                 \
          break;                                                  \
        case GrGeomOctreeFaceR:                                   \
          fdir[0] = 1; fdir[1] = 0; fdir[2] = 0;                  \
          break;                                                  \
        case GrGeomOctreeFaceD:                                   \
          fdir[0] = 0; fdir[1] = -1; fdir[2] = 0;                 \
          break;                                                  \
        case GrGeomOctreeFaceU:                                   \
          fdir[0] = 0; fdir[1] = 1; fdir[2] = 0;                  \
          break;                                                  \
        case GrGeomOctreeFaceB:                                   \
          fdir[0] = 0; fdir[1] = 0; fdir[2] = -1;                 \
          break;                                                  \
        case GrGeomOctreeFaceF:                                   \
          fdir[0] = 0; fdir[1] = 0; fdir[2] = 1;                  \
          break;                                                  \
        default:                                                  \
          fdir[0] = -9999; fdir[1] = -9999; fdir[2] = -99999;     \
          break;                                                  \
      }                                                           \
                                                                  \
      BoxArray* boxes = GrGeomSolidSurfaceBoxes(grgeom, PV_f);      \
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

#define NoWait_GrGeomSurfLoopBoxes(locals,                        \
                                   i, j, k,                       \
                                   fdir, grgeom,                  \
                                   ix, iy, iz,                    \
                                   nx, ny, nz,                    \
                                   body)                          \
  {                                                               \
    int PV_fdir[3];                                               \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;           \
    int *PV_visiting = NULL;                                      \
    fdir = PV_fdir;                                               \
    for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)       \
    {                                                             \
      switch (PV_f)                                               \
      {                                                           \
        case GrGeomOctreeFaceL:                                   \
          fdir[0] = -1; fdir[1] = 0; fdir[2] = 0;                 \
          break;                                                  \
        case GrGeomOctreeFaceR:                                   \
          fdir[0] = 1; fdir[1] = 0; fdir[2] = 0;                  \
          break;                                                  \
        case GrGeomOctreeFaceD:                                   \
          fdir[0] = 0; fdir[1] = -1; fdir[2] = 0;                 \
          break;                                                  \
        case GrGeomOctreeFaceU:                                   \
          fdir[0] = 0; fdir[1] = 1; fdir[2] = 0;                  \
          break;                                                  \
        case GrGeomOctreeFaceB:                                   \
          fdir[0] = 0; fdir[1] = 0; fdir[2] = -1;                 \
          break;                                                  \
        case GrGeomOctreeFaceF:                                   \
          fdir[0] = 0; fdir[1] = 0; fdir[2] = 1;                  \
          break;                                                  \
        default:                                                  \
          fdir[0] = -9999; fdir[1] = -9999; fdir[2] = -99999;     \
          break;                                                  \
      }                                                           \
                                                                  \
      BoxArray* boxes = GrGeomSolidSurfaceBoxes(grgeom, PV_f);      \
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
    }                                                               \
  }


#endif // _PF_OMP_GRGEOM_H
