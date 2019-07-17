
set (SRC_FILES iterativ.cpp kinsol.cpp kinspgmr.cpp llnlmath.c spgmr.cpp)

add_library(pfkinsol ${SRC_FILES})

if (${PARFLOW_HAVE_MPI})
  target_include_directories (pfkinsol PUBLIC "${MPI_C_INCLUDE_PATH}")
endif (${PARFLOW_HAVE_MPI})

install (TARGETS pfkinsol DESTINATION lib)
