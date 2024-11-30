CUDA_VER:=12.1
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

APP:= FirstApp
CXX:= g++ -std=c++17
# NVCC:=/usr/local/cuda/bin/nvcc

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)


NVDS_VERSION:=6.3

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

SRCS:= $(wildcard *.cpp)

INCS:= $(wildcard *.hpp)

PKGS:= gstreamer-1.0 opencv4

OBJS:= $(SRCS:.cpp=.o)

CFLAGS+= -I /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes \
  -I /usr/local/cuda-$(CUDA_VER)/include

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
  -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_yml_parser \
  -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) \
  -lnvbufsurface -lcublasLt -lnvbufsurftransform \

all: $(APP)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)