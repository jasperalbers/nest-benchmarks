CXX = icpc
CYTHON = cython
PYTHON = python3

# Use fancy methods to get the correct import directory
PYTHON_INC = $(shell $(PYTHON) -c 'import sys; from distutils import sysconfig; sys.stdout.write(sysconfig.get_python_inc())')
PYTHON_INC += -I$(shell $(PYTHON) -c 'import numpy; print(numpy.get_include())')

ifdef PYTHON_INC
    CPPFLAGS += -I$(PYTHON_INC)
endif

ifeq ($(CXX),g++)
    OPENMPFLAGS = -fopenmp
    OPTFLAGS = -O3 -Wall -march=native -fno-strict-aliasing -fno-associative-math
endif

ifeq ($(CXX),icpc)
    OPENMPFLAGS = -fopenmp
    OPTFLAGS = -O3 -Wall -fp-model strict -fno-strict-aliasing
endif

CXXFLAGS += -shared -DPIC -fPIC $(OPTFLAGS) $(OPENMPFLAGS)

# Put all your target modules in here
#
MODULES = _routines.pyx

all: $(MODULES:%.pyx=%.so) 

.SECONDARY: $(MODULES:%.pyx=%.cpp)

%.so: %.cpp %.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

%.cpp %.html: %.pyx
	$(CYTHON) --cplus $<

clean:
	@rm -f *.c *.cpp *.html *.o *.so *~ *.pyc core*

.PHONY: clean
