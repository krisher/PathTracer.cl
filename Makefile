
EXECUTABLE = ray-tracer
CFLAGS = -Wall
#CFLAGS = -Wall -g
CXX = g++
INSTALL = install
CPU_ARCH = x86_64

OUTPUT_DIR = bin
BUILD_DIR = build

LIBDIRS = 
LLIBS = OpenCL glut
INCLUDEDIRS = include
DOXYGEN_CFG = doxygen/doxy.cf
SRCDIR = src


CPPFILES = $(wildcard $(SRCDIR)/*.cpp)
CLFILES = src/raytracer.cl
CFLAGS += $(foreach f, $(INCLUDEDIRS), -I$(f))
LDFLAGS = $(foreach f, $(LIBDIRS), -L$(f))
LDFLAGS += $(foreach f, $(LLIBS), -l$(f))


OBJS := $(patsubst $(SRCDIR)/%,$(BUILD_DIR)/%, $(CPPFILES:.cpp=.o))
CLFILES_OUT =  $(patsubst $(SRCDIR)/%,$(OUTPUT_DIR)/%,$(CLFILES))

all : $(OUTPUT_DIR)/$(EXECUTABLE) 


$(OUTPUT_DIR)/$(EXECUTABLE) : $(OBJS) $(CLFILES_OUT) 
	@echo [$@]
	@-mkdir -p $(OUTPUT_DIR)
	$(CXX) -o $@  $(OBJS) $(LDFLAGS) 


$(OBJS): $(BUILD_DIR)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/*.h
	@echo [$@]
	@-mkdir -p $(BUILD_DIR)
	$(CXX) $(CFLAGS) -o $@ -c $<


$(CLFILES_OUT): $(OUTPUT_DIR)/% : $(SRCDIR)/%
	@-mkdir -p $(OUTPUT_DIR)
	cpp -I$(SRCDIR) < $< > $@

doc : 
	doxygen $(DOXYGEN_CFG)


clean :
	@echo [$@]
	-rm -rf $(BUILD_DIR) $(OUTPUT_DIR)

.PHONY : clean all
