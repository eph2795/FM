CC=g++
CFLAGS=-g -Wall -std=c++11 -O3 -DNDEBUG
RM=/bin/rm -f

all: main

data.o: data.h data.cpp
	${CC} ${CFLAGS} -c data.cpp

model.o: data.h regularizer.h model.h model.cpp
	${CC} ${CFLAGS} -c model.cpp

loss.o: data.h loss.h loss.cpp
	${CC} ${CFLAGS} -c loss.cpp

optimization.o: data.h model.h loss.h optimization.h optimization.cpp
	${CC} ${CFLAGS} -c optimization.cpp

regularizer.o: regularizer.h regularizer.cpp
	${CC} ${CFLAGS} -c regularizer.cpp

main.o: data.h regularizer.h model.h loss.h optimization.h main.cpp
	${CC} ${CFLAGS} -c main.cpp

main: main.o data.o model.o loss.o optimization.o regularizer.o
	${CC} ${CFLAGS} -o main main.o data.o model.o loss.o optimization.o regularizer.o

clean:
	 ${RM} *.o main
