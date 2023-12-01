# MPI C++ Compiler (Intel)
CC=mpiicc

# Compilation flags
CFLAGS=-std=c++14 -g

iterLinSolver_mpi.exe: iterlinsolver_mpi.o
	$(CC) -o $@ $^

iterlinsolver_mpi.o: iterlinsolver_mpi.cpp
	$(CC) ${CFLAGS} -c $<

clean:
	rm -r *.o *.exe
