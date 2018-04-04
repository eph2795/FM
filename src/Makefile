CC=g++
CFLAGS=-g -Wall -std=c++11
RM=/bin/rm -f

all: main

data.o: data.h data.cpp
	${CC} ${CFLAGS} -c data.cpp

main.o: data.h main.cpp
	${CC} ${CFLAGS} -c main.cpp

main: main.o data.o 
	${CC} ${CFLAGS} -o main main.o data.o

clean:
	 ${RM} *.o main
