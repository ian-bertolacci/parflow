/* Generic wrapper header file for managing different parallel implementations */

#ifndef _PF_PARALLEL_H
#define _PF_PARALLEL_H


#ifdef HAVE_OMP

#if 1
#include "pf_omploops.h"
#endif

#endif // HAVE_OMP

#endif // _PF_PARALLEL_H
