# C++ GCC compiler
CC=g++

# compilation flags (none)
LDFLAGS=-fopenmp

# source file
SOURCE=./mandelloop_omp.cpp

# binary name
EXEC=mandelLoop_omp.exe

all:
	$(CC) $(LDFLAGS) $(SOURCE) -o $(EXEC)
	@echo -e "----- COMPILATION DONE -----"


clean:
	rm -r $(EXEC)

