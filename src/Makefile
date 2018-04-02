CC=g++
CFLAGS=-g -Wall
RM=/bin/rm -f

all: main

main.o: main.cpp
	${CC} ${CFLAGS} -c main.cpp

main: main.o 
	${CC} ${CFLAGS} -o main main.o

clean:
	 ${RM} *.o main
