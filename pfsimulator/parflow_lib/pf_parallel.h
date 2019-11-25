/* Generic wrapper header file for managing different parallel implementations */

#ifndef _PF_PARALLEL_H
#define _PF_PARALLEL_H


#ifdef HAVE_OMP

/* Utility macros for inserting OMP pragmas in macros */
#define EMPTY()
#define DEFER(x) x EMPTY()
#define PRAGMA(args) _Pragma( #args )

#include "pf_omploops.h"

#endif // HAVE_OMP

#endif // _PF_PARALLEL_H
