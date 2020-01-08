#ifndef _INLINE_MGSEMI_H
#define _INLINE_MGSEMI_H

#ifdef HAVE_OMP
/* NOTE: These are intended to be used from within an OMP Parallel Region */

static inline void
Inline_Axpy(double  alpha,
     Vector *x,
     Vector *y)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *y_sub;
  Subvector  *x_sub;

  double     *yp, *xp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;

  int i_s, i, j, k, iv;


  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    y_sub = VectorSubvector(y, i_s);
    x_sub = VectorSubvector(x, i_s);

    nx_v = SubvectorNX(y_sub);
    ny_v = SubvectorNY(y_sub);
    nz_v = SubvectorNZ(y_sub);

    yp = SubvectorElt(y_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);

    iv = 0;
    __BoxLoopI1(NO_LOCALS,
               i, j, k, ix, iy, iz, nx, ny, nz,
               iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      yp[iv] += alpha * xp[iv];
    });
  }

#pragma omp single
  {
    IncFLOPCount(2 * VectorSize(x));
  }
}


static inline void
Inline_Copy(Vector *x, Vector *y)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *y_sub;
  Subvector  *x_sub;

  double     *yp, *xp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;

  int i_s, i, j, k, i_x, i_y;


  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    x_sub = VectorSubvector(x, i_s);
    y_sub = VectorSubvector(y, i_s);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    yp = SubvectorElt(y_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;
    __BoxLoopI2(NO_LOCALS,
               i, j, k, ix, iy, iz, nx, ny, nz,
               i_x, nx_x, ny_x, nz_x, 1, 1, 1,
               i_y, nx_y, ny_y, nz_y, 1, 1, 1,
    {
      yp[i_y] = xp[i_x];
    });
  }
}

static inline double
Inline_InnerProd(Vector *x,
                 Vector *y)
{
  Grid         *grid = VectorGrid(x);
  Subgrid      *subgrid;

  Subvector    *y_sub;
  Subvector    *x_sub;

  double       *yp, *xp;

  double result = 0.0;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;

  int i_s, i, j, k, iv;

  amps_Invoice result_invoice;
  #pragma omp single
  {
    result_invoice = amps_NewInvoice("%d", &result);
  }

  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    y_sub = VectorSubvector(y, i_s);
    x_sub = VectorSubvector(x, i_s);

    nx_v = SubvectorNX(y_sub);
    ny_v = SubvectorNY(y_sub);
    nz_v = SubvectorNZ(y_sub);

    yp = SubvectorElt(y_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);

    iv = 0;
    //__BoxLoopReduceI1(NO_LOCALS, result,
    __BoxLoopI1(NO_LOCALS,
                i, j, k, ix, iy, iz, nx, ny, nz,
                iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      result += yp[iv] * xp[iv];
    });
  }

  #pragma omp single
  {
    amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
    amps_FreeInvoice(result_invoice);
    IncFLOPCount(2 * VectorSize(x) - 1);
  }

  return result;
}

static inline void
Inline_Matvec(double  alpha,
              Matrix *A,
              Vector *x,
              double  beta,
              Vector *y)
{
  VectorUpdateCommHandle *handle = NULL;

  Grid           *grid = MatrixGrid(A);
  Subgrid        *subgrid;

  SubregionArray *subregion_array;
  Subregion      *subregion;

  ComputePkg     *compute_pkg;

  Region         *compute_reg = NULL;

  Subvector      *y_sub = NULL;
  Subvector      *x_sub = NULL;
  Submatrix      *A_sub = NULL;

  Stencil        *stencil;
  int stencil_size;
  StencilElt     *s;

  int compute_i, sg, sra, sr, si, i, j, k;

  double temp;

  double         *ap;
  double         *xp;
  double         *yp;

  int vi, mi;

  int ix, iy, iz;
  int nx, ny, nz;
  int sx, sy, sz;

  int nx_v = 0, ny_v = 0, nz_v = 0;
  int nx_m = 0, ny_m = 0, nz_m = 0;

  /*-----------------------------------------------------------------------
   * Begin timing
   *-----------------------------------------------------------------------*/

#pragma omp single
  {
    BeginTiming(MatvecTimingIndex);
#ifdef VECTOR_UPDATE_TIMING
    EventTiming[NumEvents][MatvecStart] = amps_Clock();
#endif
  }

  /*-----------------------------------------------------------------------
   * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
   *-----------------------------------------------------------------------*/

  if (alpha == 0.0)
  {
    ForSubgridI(sg, GridSubgrids(grid))
    {
      subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      if (nx && ny && nz)
      {
        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        y_sub = VectorSubvector(y, sg);

        nx_v = SubvectorNX(y_sub);
        ny_v = SubvectorNY(y_sub);
        nz_v = SubvectorNZ(y_sub);

        yp = SubvectorElt(y_sub, ix, iy, iz);

        int vi = 0;
        __BoxLoopI1(NO_LOCALS,
                   i, j, k,
                   ix, iy, iz, nx, ny, nz,
                   vi, nx_v, ny_v, nz_v, 1, 1, 1,
        {
          yp[vi] *= beta;
        });
      }
    }

    #pragma omp single
    {
      IncFLOPCount(VectorSize(x));
      EndTiming(MatvecTimingIndex);
    }

    return;
  }

  /*-----------------------------------------------------------------------
   * Do (alpha != 0.0) computation
   *-----------------------------------------------------------------------*/

  compute_pkg = GridComputePkg(grid, VectorUpdateAll);

  //int vi;
  //int mi;
  //int tid = omp_get_thread_num();

  for (compute_i = 0; compute_i < 2; compute_i++)
  {
    switch (compute_i)
    {
      case 0:

#pragma omp master
      {
#ifndef NO_VECTOR_UPDATE
#ifdef VECTOR_UPDATE_TIMING
        BeginTiming(VectorUpdateTimingIndex);
        EventTiming[NumEvents][InitStart] = amps_Clock();
#endif
        handle = InitVectorUpdate(x, VectorUpdateAll);

#ifdef VECTOR_UPDATE_TIMING
        EventTiming[NumEvents][InitEnd] = amps_Clock();
        EndTiming(VectorUpdateTimingIndex);
#endif
#endif
      }

      compute_reg = ComputePkgIndRegion(compute_pkg);
      /*-----------------------------------------------------------------
       * initialize y= (beta/alpha)*y
       *-----------------------------------------------------------------*/

      ForSubgridI(sg, GridSubgrids(grid))
      {
        subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        if (nx && ny && nz)
        {
          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          y_sub = VectorSubvector(y, sg);

          nx_v = SubvectorNX(y_sub);
          ny_v = SubvectorNY(y_sub);
          nz_v = SubvectorNZ(y_sub);


          if (beta != alpha)
          {
            temp = beta / alpha;
            yp = SubvectorElt(y_sub, ix, iy, iz);
            vi = 0;
            __BoxLoopI1(NO_LOCALS,
                        i, j, k,
                        ix, iy, iz, nx, ny, nz,
                        vi, nx_v, ny_v, nz_v, 1, 1, 1,
            {
              yp[vi] *= temp;
            });
          }
/*
          if (temp != 1.0)
          {

            if (temp == 0.0)
            {
              __BoxLoopI1(NO_LOCALS,
                          i, j, k,
                          ix, iy, iz, nx, ny, nz,
                          vi, nx_v, ny_v, nz_v, 1, 1, 1,
              {
                yp[vi] = 0.0;
              });
            }
            else
            {
              __BoxLoopI1(NO_LOCALS,
                          i, j, k,
                          ix, iy, iz, nx, ny, nz,
                          vi, nx_v, ny_v, nz_v, 1, 1, 1,
              {
                yp[vi] *= temp;
              });
            }
          }
*/
        }
      }

      break;

      case 1:

#pragma omp master
      {
#ifndef NO_VECTOR_UPDATE
#ifdef VECTOR_UPDATE_TIMING
        BeginTiming(VectorUpdateTimingIndex);
        EventTiming[NumEvents][FinalizeStart] = amps_Clock();
#endif
        FinalizeVectorUpdate(handle);

#ifdef VECTOR_UPDATE_TIMING
        EventTiming[NumEvents][FinalizeEnd] = amps_Clock();
        EndTiming(VectorUpdateTimingIndex);
#endif
#endif
      }

      compute_reg = ComputePkgDepRegion(compute_pkg);
      break;
    }

    ForSubregionArrayI(sra, compute_reg)
    {
      subregion_array = RegionSubregionArray(compute_reg, sra);

      if (SubregionArraySize(subregion_array))
      {
        y_sub = VectorSubvector(y, sra);
        x_sub = VectorSubvector(x, sra);

        A_sub = MatrixSubmatrix(A, sra);

        nx_v = SubvectorNX(y_sub);
        ny_v = SubvectorNY(y_sub);
        nz_v = SubvectorNZ(y_sub);

        nx_m = SubmatrixNX(A_sub);
        ny_m = SubmatrixNY(A_sub);
        nz_m = SubmatrixNZ(A_sub);
      }

      /*-----------------------------------------------------------------
       * y += A*x
       *-----------------------------------------------------------------*/

      ForSubregionI(sr, subregion_array)
      {
        subregion = SubregionArraySubregion(subregion_array, sr);

        ix = SubregionIX(subregion);
        iy = SubregionIY(subregion);
        iz = SubregionIZ(subregion);

        nx = SubregionNX(subregion);
        ny = SubregionNY(subregion);
        nz = SubregionNZ(subregion);

        sx = SubregionSX(subregion);
        sy = SubregionSY(subregion);
        sz = SubregionSZ(subregion);

        stencil = MatrixStencil(A);
        stencil_size = StencilSize(stencil);
        s = StencilShape(stencil);

        yp = SubvectorElt(y_sub, ix, iy, iz);

        for (si = 0; si < stencil_size; si++)
        {
          xp = SubvectorElt(x_sub,
                            (ix + s[si][0]),
                            (iy + s[si][1]),
                            (iz + s[si][2]));
          ap = SubmatrixElt(A_sub, si, ix, iy, iz);

          vi = 0; mi = 0;
          __BoxLoopI2(NO_LOCALS,
                      i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      vi, nx_v, ny_v, nz_v, sx, sy, sz,
                      mi, nx_m, ny_m, nz_m, 1, 1, 1,
          {
            yp[vi] += ap[mi] * xp[vi];
          });
        }

        if (alpha != 1.0)
        {
          yp = SubvectorElt(y_sub, ix, iy, iz);

          vi = 0;
          __BoxLoopI1(NO_LOCALS,
                      i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      vi, nx_v, ny_v, nz_v, 1, 1, 1,
          {
            yp[vi] *= alpha;
          });
        }
      }
    }
  }
  /*-----------------------------------------------------------------------
   * End timing
   *-----------------------------------------------------------------------*/
#pragma omp master
  {
    IncFLOPCount(2 * (MatrixSize(A) + VectorSize(x)));
    EndTiming(MatvecTimingIndex);

#ifdef VECTOR_UPDATE_TIMING
    EventTiming[NumEvents++][MatvecEnd] = amps_Clock();
#endif
  }
}


static inline void
Inline_MGSemiRestrict(Matrix *        A_f,
                      Vector *        r_f,
                      Vector *        r_c,
                      Matrix *        P,
                      SubregionArray *f_sr_array,
                      SubregionArray *c_sr_array,
                      ComputePkg *    compute_pkg,
                      CommPkg *       r_f_comm_pkg)
{
  SubregionArray *subregion_array;

  Subregion      *subregion;

  Region         *compute_reg = NULL;

  Subvector      *r_f_sub;
  Subvector      *r_c_sub;

  Submatrix      *P_sub;

  StencilElt     *s;

  double         *r_fp, *r_cp;

  double         *p1, *p2;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_p, ny_p, nz_p;
  int nx_f, ny_f, nz_f;
  int nx_c, ny_c, nz_c;

  int i_p, i_f, i_c;

  int sx, sy, sz;

  int compute_i, i, j, ii, jj, kk;
  int stride;

  CommHandle     *handle = NULL;

  (void)A_f;
  (void)f_sr_array;
  (void)c_sr_array;

  /*--------------------------------------------------------------------
   * Compute r_c in c_sr_array
   *--------------------------------------------------------------------*/

  for (compute_i = 0; compute_i < 2; compute_i++)
  {
    switch (compute_i)
    {
      case 0:
#pragma omp master
      {
        handle = InitCommunication(r_f_comm_pkg);
      }
      compute_reg = ComputePkgIndRegion(compute_pkg);
      break;

      case 1:
#pragma omp master
      {
        FinalizeCommunication(handle);
      }
      compute_reg = ComputePkgDepRegion(compute_pkg);
      break;
    }

    ForSubregionArrayI(i, compute_reg)
    {
      subregion_array = RegionSubregionArray(compute_reg, i);

      if (SubregionArraySize(subregion_array))
      {
        r_c_sub = VectorSubvector(r_c, i);
        r_f_sub = VectorSubvector(r_f, i);

        P_sub = MatrixSubmatrix(P, i);

        nx_p = SubmatrixNX(P_sub);
        ny_p = SubmatrixNY(P_sub);
        nz_p = SubmatrixNZ(P_sub);

        nx_c = SubvectorNX(r_c_sub);
        ny_c = SubvectorNY(r_c_sub);
        nz_c = SubvectorNZ(r_c_sub);

        nx_f = SubvectorNX(r_f_sub);
        ny_f = SubvectorNY(r_f_sub);
        nz_f = SubvectorNZ(r_f_sub);
      }

      ForSubregionI(j, subregion_array)
      {
        subregion = SubregionArraySubregion(subregion_array, j);

        ix = SubregionIX(subregion);
        iy = SubregionIY(subregion);
        iz = SubregionIZ(subregion);

        nx = SubregionNX(subregion);
        ny = SubregionNY(subregion);
        nz = SubregionNZ(subregion);

        sx = SubregionSX(subregion);
        sy = SubregionSY(subregion);
        sz = SubregionSZ(subregion);

        stride = (sx - 1) + ((sy - 1) + (sz - 1) * ny_f) * nx_f;

        r_cp = SubvectorElt(r_c_sub, ix / sx, iy / sy, iz / sz);

        r_fp = SubvectorElt(r_f_sub, ix, iy, iz);

        s = StencilShape(MatrixStencil(P));

        p1 = SubmatrixElt(P_sub, 1,
                          (ix + s[0][0]), (iy + s[0][1]), (iz + s[0][2]));
        p2 = SubmatrixElt(P_sub, 0,
                          (ix + s[1][0]), (iy + s[1][1]), (iz + s[1][2]));

        i_p = 0;
        i_c = 0;
        i_f = 0;
        __BoxLoopI3(NO_LOCALS,
                    ii, jj, kk, ix, iy, iz, nx, ny, nz,
                    i_p, nx_p, ny_p, nz_p, 1, 1, 1,
                    i_c, nx_c, ny_c, nz_c, 1, 1, 1,
                    i_f, nx_f, ny_f, nz_f, sx, sy, sz,
        {
          r_cp[i_c] = r_fp[i_f] + (p1[i_p] * r_fp[i_f - stride] +
                                   p2[i_p] * r_fp[i_f + stride]);
        });
      }
    }
  }
  /*-----------------------------------------------------------------------
   * Increment the flop counter
   *-----------------------------------------------------------------------*/
#pragma omp single
  {
    IncFLOPCount(3 * VectorSize(r_c));
  }
}

static inline void
Inline_MGSemiProlong(
                     Matrix *        A_f,
                     Vector *        e_f,
                     Vector *        e_c,
                     Matrix *        P,
                     SubregionArray *f_sr_array,
                     SubregionArray *c_sr_array,
                     ComputePkg *    compute_pkg,
                     CommPkg *       e_f_comm_pkg)
{
  SubregionArray *subregion_array;

  Subregion      *subregion;

  Region         *compute_reg = NULL;

  Subvector      *e_f_sub;
  Subvector      *e_c_sub;

  Submatrix      *P_sub;

  double         *e_fp, *e_cp;

  double         *p1, *p2;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_f, ny_f, nz_f;
  int nx_c, ny_c, nz_c;

  int i_f, i_c;

  int sx, sy, sz;

  int compute_i, i, j, ii, jj, kk;
  int stride;

  CommHandle     *handle = NULL;

  (void)A_f;
  (void)f_sr_array;

  /*--------------------------------------------------------------------
   * Compute e_f in c_sr_array
   *--------------------------------------------------------------------*/
  ForSubregionI(i, c_sr_array)
  {
    subregion = SubregionArraySubregion(c_sr_array, i);

    ix = SubregionIX(subregion);
    iy = SubregionIY(subregion);
    iz = SubregionIZ(subregion);

    nx = SubregionNX(subregion);
    ny = SubregionNY(subregion);
    nz = SubregionNZ(subregion);

    if (nx && ny && nz)
    {
      sx = SubregionSX(subregion);
      sy = SubregionSY(subregion);
      sz = SubregionSZ(subregion);

      e_c_sub = VectorSubvector(e_c, i);
      e_f_sub = VectorSubvector(e_f, i);

      nx_c = SubvectorNX(e_c_sub);
      ny_c = SubvectorNY(e_c_sub);
      nz_c = SubvectorNZ(e_c_sub);

      nx_f = SubvectorNX(e_f_sub);
      ny_f = SubvectorNY(e_f_sub);
      nz_f = SubvectorNZ(e_f_sub);

      e_cp = SubvectorElt(e_c_sub, ix / sx, iy / sy, iz / sz);
      e_fp = SubvectorElt(e_f_sub, ix, iy, iz);

      i_c = 0;
      i_f = 0;
      __BoxLoopI2(NO_LOCALS,
                 ii, jj, kk, ix, iy, iz, nx, ny, nz,
                 i_c, nx_c, ny_c, nz_c, 1, 1, 1,
                 i_f, nx_f, ny_f, nz_f, sx, sy, sz,
      {
        e_fp[i_f] = e_cp[i_c];
      });
    }
  }

  /*--------------------------------------------------------------------
   * Compute e_f in f_sr_array
   *--------------------------------------------------------------------*/

  for (compute_i = 0; compute_i < 2; compute_i++)
  {
    switch (compute_i)
    {
      case 0:
#pragma omp master
      {
        handle = InitCommunication(e_f_comm_pkg);
      }
        compute_reg = ComputePkgIndRegion(compute_pkg);
        break;

      case 1:
#pragma omp master
      {
        FinalizeCommunication(handle);
      }
        compute_reg = ComputePkgDepRegion(compute_pkg);
        break;
    }

    ForSubregionArrayI(i, compute_reg)
    {
      subregion_array = RegionSubregionArray(compute_reg, i);

      if (SubregionArraySize(subregion_array))
      {
        e_f_sub = VectorSubvector(e_f, i);

        P_sub = MatrixSubmatrix(P, i);

        nx_f = SubvectorNX(e_f_sub);
        ny_f = SubvectorNY(e_f_sub);
        nz_f = SubvectorNZ(e_f_sub);

        nx_c = SubmatrixNX(P_sub);
        ny_c = SubmatrixNY(P_sub);
        nz_c = SubmatrixNZ(P_sub);
      }

      ForSubregionI(j, subregion_array)
      {
        subregion = SubregionArraySubregion(subregion_array, j);

        ix = SubregionIX(subregion);
        iy = SubregionIY(subregion);
        iz = SubregionIZ(subregion);

        nx = SubregionNX(subregion);
        ny = SubregionNY(subregion);
        nz = SubregionNZ(subregion);

        sx = SubregionSX(subregion);
        sy = SubregionSY(subregion);
        sz = SubregionSZ(subregion);

        stride = (sx - 1) + ((sy - 1) + (sz - 1) * ny_f) * nx_f;

        e_fp = SubvectorElt(e_f_sub, ix, iy, iz);

        p1 = SubmatrixElt(P_sub, 0, ix, iy, iz);
        p2 = SubmatrixElt(P_sub, 1, ix, iy, iz);

        i_c = 0;
        i_f = 0;
        __BoxLoopI2(NO_LOCALS,
                   ii, jj, kk, ix, iy, iz, nx, ny, nz,
                   i_c, nx_c, ny_c, nz_c, 1, 1, 1,
                   i_f, nx_f, ny_f, nz_f, sx, sy, sz,
        {
          e_fp[i_f] = (p1[i_c] * e_fp[i_f - stride] +
                       p2[i_c] * e_fp[i_f + stride]);
        });
      }
    }
  }
  /*-----------------------------------------------------------------------
   * Increment the flop counter
   *-----------------------------------------------------------------------*/
#pragma omp single
  {
    IncFLOPCount(3 * VectorSize(e_c));
  }
}

#else // #ifndef USING_PARALLEL && HAVE_OMP

#define Inline_Axpy(alpha, x, y) Axpy(alpha, x, y)
#define Inline_Copy(x, y) Copy(x, y)
#define Inline_InnerProd(x, y) InnerProd(x, y)
#define Inline_Matvec(alpha, A, x, beta, y) Matvec(alpha, A, x, beta, y)
#define Inline_MGSemiRestrict(A_f, r_f, r_c, P, f_sr_array, c_sr_array, compute_pkg, r_f_comm_pkg) \
  MGSemiRestrict(A_f, r_f, r_c, P, f_ar_array, c_sr_array, compute_pkg, r_f_comm_pkg)
#define Inline_MGSemiProlong(A_f, e_f, e_c, P, f_sr_array, c_sr_array, compute_pkg, e_f_comm_pkg) \
  MGSemiProlong(A-f, e_f, e_c, P, f_sr_array, c_sr_array, compute_pkg, e_f_comm_pkg)

#endif

#endif // _INLINE_MGSEMI_H
