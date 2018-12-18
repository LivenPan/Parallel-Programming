CXX = g++
MPICXX = mpicxx
CFLAGS  = -Wall
LDFLAGS = -std=c++11 -O3 -lpthread -lm -fopenmp 
CXXFLAGS = $(CFLAGS) 

all: APSP_Pthread APSP_MPI_sync APSP_MPI_async

APSP_MPI_sync : APSP_MPI_sync.cc
	$(MPICXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?

APSP_MPI_async : APSP_MPI_async.cc
	$(MPICXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?

APSP_Pthread: APSP_Pthread.cc
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?

clean:
	rm -f APSP_MPI_sync APSP_Pthread APSP_MPI_async
