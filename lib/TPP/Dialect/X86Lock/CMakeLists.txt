add_mlir_dialect_library(TPPX86LockDialect
  # Ops and dialects
    X86LockDialect.cpp
    X86LockOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    # add_mlir_dialect macro force-prefixes with MLIR
    MLIRX86LockOpsIncGen
    TPPCompilerPassIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(TPPX86LockDialect
  PUBLIC
    $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
