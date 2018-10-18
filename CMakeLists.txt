cmake_minimum_required(VERSION 2.8.4)
project(codigo)

set (LIBSVM_DIR "externals/libsvm")
set (LIBNIFTI_DIR "externals/libnifti")
set (LIBJPEG_DIR "externals/libjpeg")
set (LIBPNG_DIR "externals/libpng")
set (TSNE_DIR "externals/tsne")

#set (LIBIFT_INC "-I ./include")
#set (LIBIFT_LD "-L ./lib -lift")
#set (LIBSVM_INC   "-I $(LIBSVM_DIR)/include")
#set (LIBSVM_LD    "-L $(LIBSVM_DIR)/lib -lsvm -lstdc++")
#set (LIBCBLAS_INC "-I /usr/local/opt/openblas/include")
#set (LIBCBLAS_LD  "-L /usr/local/opt/openblas/lib -L /lib64/atlas-sse3 -L /usr/lib/atlas-base -llapack -lblas -lcblas")

#set (LIBNIFTI_INC "-I $(LIBNIFTI_DIR)/include")
#set (LIBJPEG_INC "-I $(LIBJPEG_DIR)/include")
#set (TSNE_INC "-I $(TSNE_DIR)/include")

#EXTERNALS_LD = -fopenmp -lm -lpng -lz

#INCLUDES = $(LIBIFT_INC) $(LIBSVM_INC) $(LIBCBLAS_INC) $(LIBNIFTI_INC) $(LIBJPEG_INC) $(TSNE_INC)
#LIBS     = $(LIBIFT_LD) $(LIBSVM_LD) $(LIBCBLAS_LD) $(EXTERNALS_LD)

#add_executable(CNNlayer CNNlayer.c)
add_executable(training training.c)
add_executable(testing testing.c)

include_directories (
        "${LIBSVM_DIR}/include"
        "include"
        "${LIBNIFTI_DIR}/include"
        "${LIBJPEG_DIR}/include"
        "${TSNE_DIR}/include"
)

set (EXTRA_LIBS ${LIBSVM_DIR}/lib lib ${LIBNIFTI_DIR}/lib ${LIBJPEG_DIR}/lib ${TSNE_DIR}/lib)
