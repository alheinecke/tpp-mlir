add_mlir_conversion_library(TPPX86LockToMemRef
  X86LockToMemRef.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTransforms
    TPPX86LockDialect
)

target_include_directories(TPPX86LockToMemRef
  PUBLIC
    $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
