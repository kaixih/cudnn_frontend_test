INCLUDES = -I cudnn-frontend/include/
CPPFLAGS = $(INCLUDES) -DNV_CUDNN_DISABLE_EXCEPTION -lcudnn -std=c++17
CXX = nvcc

SRCS := $(wildcard test_*.cpp)
OBJS = $(SRCS:.cpp=.out)

all: $(OBJS)

$(OBJS): %.out: %.cpp
	${CXX} ${CPPFLAGS} -o $@ $<

clean:
	rm -rf *.out

