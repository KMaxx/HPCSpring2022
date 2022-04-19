EXECUTABLES = vec_inn mat_inn jacobi2D 
COMPILER = nvcc
FLAGS = -arch=sm_61 -Xcompiler -fopenmp

all: $(EXECUTABLES)


vec_inn: ./vec_inn.cu 
	nvcc -arch=sm_61 vec_inn.cu -o vec_inn -Xcompiler -fopenmp

run_vec_inn: vec_inn
	./vec_inn

mat_inn: ./mat_inn.cu 
	nvcc -arch=sm_61 mat_inn.cu -o mat_inn -Xcompiler -fopenmp

run_mat_inn: mat_inn
	./mat_inn

jacobi2D: ./jacobi2D.cu 
	nvcc -arch=sm_61 jacobi2D.cu -o jacobi2D -Xcompiler -fopenmp

run_jacobi: jacobi2D
	./jacobi2D

clean:
	rm -rf $(EXECUTABLES)
