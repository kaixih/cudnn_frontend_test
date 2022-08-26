INCLUDES = -I cudnn-frontend/include/ -I src/
CPPFLAGS = $(INCLUDES) -DNV_CUDNN_DISABLE_EXCEPTION -std=c++17
LDFLAGS = -lcudnn
CXX = nvcc
HEADERS = $(wildcard src/*.h)
SOURCES = $(wildcard src/*.cc)

EXEC = run_conv_graphs.out run_matmul_graphs.out
OBJECTS = $(SOURCES:.cc=.o)

all: ${EXEC}

%.o: %.cc ${HEADERS}
	${CXX} ${CPPFLAGS} -c -o $@ $<

run_conv_graphs.out: samples/run_conv_graphs.cc ${OBJECTS}
	${CXX} ${CPPFLAGS} ${LDFLAGS} -o $@ $^

run_matmul_graphs.out: samples/run_matmul_graphs.cc ${OBJECTS}
	${CXX} ${CPPFLAGS} ${LDFLAGS} -o $@ $^

clean:
	rm -rf ${EXEC} ${OBJECTS}

