# C++ GCC compiler
CC=clang++-mp-16

# compilation flags (none)
LDFLAGS=-fopenmp

# source file
SOURCE=./nestedloopmapinsert_omp.cpp

# binary name
EXEC=nestedLoopMapInsert_omp

all:
	$(CC) $(LDFLAGS) $(SOURCE) -o $(EXEC)
	@echo -e "----- COMPILATION DONE -----"


clean:
	rm -r $(EXEC)

