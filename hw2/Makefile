EXECUTABLES = MMult1 val_test01_solved val_test02_solved omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp
COMPILER = g++
FLAGS = -fopenmp

all: $(EXECUTABLES)

MMult1: MMult1.cpp 
	g++ -O3 -std=gnu++11 -o MMult1 -fopenmp -march=native MMult1.cpp

run_MMult1: MMult1 
	./MMult1

val_test01_solved: val_test01_solved.cpp
	g++ -o val_test01_solved -g val_test01_solved.cpp

run_val_test01_solved: val_test01_solved
	valgrind ./val_test01_solved

val_test02_solved: val_test02_solved.cpp
	g++ -o val_test02_solved -g val_test02_solved.cpp

run_val_test02_solved: val_test02_solved
	valgrind ./val_test02_solved

omp_solved2: omp_solved2.c 
	gcc -o omp_solved2 -fopenmp omp_solved2.c

solved2: omp_solved2 
	./omp_solved2

omp_solved3: omp_solved3.c 
	gcc -o omp_solved3 -fopenmp omp_solved3.c

solved3: omp_solved3
	./omp_solved3

omp_solved4: omp_solved4.c 
	gcc -o omp_solved4 -fopenmp omp_solved4.c

solved4: omp_solved4
	./omp_solved4

omp_solved5: omp_solved5.c 
	gcc -o omp_solved5 -fopenmp omp_solved5.c

solved5: omp_solved5
	./omp_solved5

omp_solved6: omp_solved6.c 
	gcc -o omp_solved6 -fopenmp omp_solved6.c

solved6: omp_solved6
	./omp_solved6

jacobi: ./jacobi2D-omp.cpp 
	g++ -O3 -std=gnu++11 -o jacobi2D-omp -fopenmp jacobi2D-omp.cpp

run_jacobi: jacobi2D-omp
	./jacobi2D-omp

gs: ./gs2D-omp.cpp 
	g++ -O3 -std=gnu++11 -o gs2D-omp -fopenmp gs2D-omp.cpp

run_gs: gs2D-omp
	./gs2D-omp

clean:
	rm -rf $(EXECUTABLES)
