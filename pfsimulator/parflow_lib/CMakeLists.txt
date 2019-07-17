set (SRC_FILES advect.F
  advection_godunov.cpp
  axpy.cpp
  background.cpp
  bc_lb.cpp
  bc_pressure.cpp
  bc_pressure_package.cpp
  calc_elevations.cpp
  cghs.cpp
  char_vector.cpp
  chebyshev.cpp
  comm_pkg.cpp
  communication.cpp
  computation.cpp
  compute_maximums.cpp
  compute_total_concentration.cpp
  constantRF.cpp
  constant_porosity.cpp
  copy.cpp
  create_grid.cpp
  diag_scale.cpp
  diffuse_lb.cpp
  discretize_pressure.cpp
  distribute_usergrid.cpp
  dpofa.cpp
  dposl.cpp
  gauinv.cpp
  general.cpp
  geom_t_solid.cpp
  geometry.cpp
  globals.cpp
  grgeom_list.cpp
  grgeom_octree.cpp
  grgeometry.cpp
  grid.cpp
  hbt.cpp
  infinity_norm.cpp
  innerprod.cpp
  input_porosity.cpp
  inputRF.cpp
  input_database.cpp
  kinsol_nonlin_solver.cpp
  kinsol_pc.cpp
  l2_error_norm.cpp
  line_process.cpp
  logging.cpp
  matdiag_scale.cpp
  matrix.cpp
  matvec.cpp
  max_field_value.cpp
  mg_semi.cpp
  mg_semi_prolong.cpp
  mg_semi_restrict.cpp
  new_endpts.cpp
  nodiag_scale.cpp
  pcg.cpp
  permeability_face.cpp
  perturb_lb.cpp
  pf_module.cpp
  pfield.cpp
  pf_pfmg.cpp
  pf_pfmg_octree.cpp
  pf_smg.cpp
  pgsRF.cpp
  phase_velocity_face.cpp
  ppcg.cpp
  nl_function_eval.cpp
  printgrid.cpp
  printmatrix.cpp
  printvector.cpp
  problem.cpp
  problem_bc.cpp
  problem_bc_internal.cpp
  problem_bc_phase_saturation.cpp
  problem_bc_pressure.cpp
  problem_capillary_pressure.cpp
  problem_domain.cpp
  problem_eval.cpp
  problem_geometries.cpp
  problem_phase_heat_capacity.cpp
  problem_ic_phase_concen.cpp
  problem_ic_phase_pressure.cpp
  problem_ic_phase_satur.cpp
  problem_mannings.cpp
  problem_phase_density.cpp
  problem_phase_internal_energy.cpp
  problem_phase_mobility.cpp
  problem_phase_rel_perm.cpp
  problem_phase_source.cpp
  problem_phase_viscosity.cpp
  problem_porosity.cpp
  problem_real_space_z.cpp
  problem_retardation.cpp
  problem_richards_bc_internal.cpp
  problem_saturation.cpp
  problem_thermal_conductivity.cpp
  problem_saturation_constitutive.cpp
  problem_spec_storage.cpp
  problem_dz_scale.cpp
  problem_toposlope_x.cpp
  problem_toposlope_y.cpp
  process_grid.cpp
  random.cpp
  ratqr.cpp
  rb_GS_point.cpp
  read_parflow_binary.cpp
  read_parflow_netcdf.cpp
  reg_from_stenc.cpp
  region.cpp
  richards_jacobian_eval.cpp
  sadvect.F
  sadvection_godunov.cpp
  scale.cpp
  select_time_step.cpp
  set_problem_data.cpp
  sim_shear.cpp
  solver.cpp
  solver_impes.cpp
  solver_lb.cpp
  solver_richards.cpp
  subsrf_sim.cpp
  time_cycle_data.cpp
  timing.cpp
  total_velocity_face.cpp
  turning_bandsRF.cpp
  usergrid_input.cpp
  vector.cpp
  vector_utilities.cpp
  w_jacobi.cpp
  well.cpp
  well_package.cpp
  wells_lb.cpp
  write_parflow_binary.cpp
  write_parflow_netcdf.cpp
  write_clm_netcdf.cpp
  write_parflow_silo.cpp
  write_parflow_silo_pmpio.cpp
  wrf_parflow.cpp
  n_vector.cpp
  nl_function_eval.cpp
  evaptranssum.cpp
  overlandsum.cpp
  overlandflow_eval.cpp
  overlandflow_eval_diffusive.cpp
  compute_top.cpp
  input_checks.cpp
  )

add_library(pfsimulator ${SRC_FILES})

if (${PARFLOW_HAVE_MPI})
  target_include_directories (pfsimulator PUBLIC "${MPI_C_INCLUDE_PATH}")
endif (${PARFLOW_HAVE_MPI})


install (TARGETS pfsimulator DESTINATION lib)

if (${PARFLOW_HAVE_HYPRE})
  target_include_directories (pfsimulator PUBLIC ${HYPRE_INCLUDE_DIR})
endif (${PARFLOW_HAVE_HYPRE})

if (${PARFLOW_HAVE_MPI})
  target_include_directories (pfsimulator PUBLIC "${MPI_C_INCLUDE_PATH}")
endif (${PARFLOW_HAVE_MPI})

if (${PARFLOW_HAVE_HDF5})
  target_include_directories (pfsimulator PUBLIC "${HDF5_INCLUDE_DIRS}")
endif (${PARFLOW_HAVE_HDF5})

if (${PARFLOW_HAVE_SILO})
  target_include_directories (pfsimulator PUBLIC "${SILO_INCLUDE_DIRS}")
endif (${PARFLOW_HAVE_SILO})

if (${PARFLOW_HAVE_NETCDF})
  target_include_directories (pfsimulator PUBLIC "${netCDF_INCLUDE_DIRS}")
  target_include_directories (pfsimulator PUBLIC "${NETCDF_INCLUDE_DIRS}")
endif (${PARFLOW_HAVE_NETCDF})
