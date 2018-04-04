CC=g++
CFLAGS=-g -Wall -std=c++11
RM=/bin/rm -f

all: main

data.o: data.h data.cpp
	${CC} ${CFLAGS} -c data.cpp

optimization.o: data.h optimization.h optimization.cpp
	${CC} ${CFLAGS} -c optimization.cpp

main.o: data.h optimization.h main.cpp
	${CC} ${CFLAGS} -c main.cpp

main: main.o data.o optimization.o
	${CC} ${CFLAGS} -o main main.o data.o optimization.o

clean:
	 ${RM} *.o main
