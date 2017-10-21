CC = mpicc
CXX = mpicxx
CFLAGS = -Wall -O3 -std=gnu99
CXXFLAGS = -Wall -O3 -std=c++11

STUDENTID = $(USER:p%=%)
TARGET1 = HW1_$(STUDENTID)_basic
TARGET2 = HW1_$(STUDENTID)_advanced
TARGET3 = Source1
SOURCE1 = HW1_$(STUDENTID)_basic.cc
SOURCE2 = HW1_$(STUDENTID)_advanced.cc
SOURCE3 = Source1.cpp
OBJECT1 = $(SOURCE1:.cc=.o)
OBJECT2 = $(SOURCE2:.cc=.o)
OBJECT3 = $(SOURCE3:.cpp=.o)


#.PHONY: all
#all: $(SOURCES) $(TARGETS)

$(TARGET1): $(OBJECT1)
	$(CXX) $(CXXFLAGS) $(OBJECT1) -o $@
$(TARGET2): $(OBJECT2)
	$(CXX) $(CXXFLAGS) $(OBJECT2) -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@
	
#.PHONY: clean
clean: 
	rm -f $(TARGET1) $(TARGET2) .o ./output/.out *.out