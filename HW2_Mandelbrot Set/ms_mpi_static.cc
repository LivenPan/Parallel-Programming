#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <assert.h>
#include <iostream>

double x0 = 0, y0 = 0;
double left = 0, right = 0;	        //real-axis min & max
double lower = 0, upper = 0;		//image-axis min &max
int num_threads = 0, width = 0, height = 0;
const char* filename;
int rankID = 0;						//Identify the process.
int numOfProcess = 0;				//Total number of process.
int dataOfperProcess = 0;  			//data amount per process has
int jobOfheight = 0;
int *slaveBuffer;
int *masterBuffer;
double start,end;

void write_png(const char* filename, const int width, const int height, const int* buffer) {
	FILE* fp = fopen(filename, "wb");
	assert(fp);
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	assert(png_ptr);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	assert(info_ptr);
	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_write_info(png_ptr, info_ptr);
	size_t row_size = 3 * width * sizeof(png_byte);
	png_bytep row = (png_bytep)malloc(row_size);
	for (int y = 0; y < height; ++y) {
		memset(row, 0, row_size);
		for (int x = 0; x < width; ++x) {
			int p = buffer[(height - 1 - y) * width + x];
			row[x * 3] = ((p & 0xf) << 4);
		}
		png_write_row(png_ptr, row);
	}
	free(row);
	png_write_end(png_ptr, NULL);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(fp);
}

void mandelbrotSet(int jobSize) {
	for (int i = 0; i < width; i++) {
		x0 = i * ((right - left) / width) + left;
		for (int j = 0; j < jobSize; j++) {
			y0 = (j+(jobOfheight * rankID))* ((upper - lower) / height) + lower;
			int repeats = 0;
			double x = 0;
			double y = 0;
			double length_squared = 0.0;
			while (repeats < 100000 && length_squared < 4.0) {
				double temp = x * x - y * y + x0;
				y = 2 * x * y + y0;
				x = temp;
				length_squared = x * x + y * y;
				++repeats;
			}
			slaveBuffer[j * width + i] = repeats;
		}
	}
}

int main(int argc, char** argv) {

	//Check input legality.
	if (argc != 9) {
		printf("Please obey the following input format: num_threads left right lower upper width height filename\n");
		return -1;
	}

	num_threads = strtol(argv[1], 0, 10);
	left = strtod(argv[2], 0);
	right = strtod(argv[3], 0);
	lower = strtod(argv[4], 0);
	upper = strtod(argv[5], 0);
	width = strtol(argv[6], 0, 10);
	height = strtol(argv[7], 0, 10);
	filename = argv[8];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
	MPI_Comm_rank(MPI_COMM_WORLD, &rankID);

	/*Calculate every job size by width&height*/
	jobOfheight = height / numOfProcess;
	
	/*Handle remaining data*/
	if(height % numOfProcess)	
		jobOfheight++;
		
	dataOfperProcess = jobOfheight * width;	
	
	slaveBuffer = new int[dataOfperProcess];	// Construct memory space for every process
		
	start = MPI_Wtime();
	mandelbrotSet(jobOfheight);	
	end = MPI_Wtime();
		
	/*Master collect data*/
	if (rankID == 0) {
		masterBuffer = new int[numOfProcess * dataOfperProcess];
	}

	MPI_Gather(slaveBuffer, dataOfperProcess, MPI_INT, masterBuffer, dataOfperProcess, MPI_INT, 0, MPI_COMM_WORLD);
		
	/* Image plot*/
	if(rankID ==0){
		write_png(filename, width, height, masterBuffer);
	}
	
	std::cout<<"rank id :" <<rankID << "Computation time :" << end-start << std::endl;

	MPI_Finalize();

	delete [] slaveBuffer;
	delete [] masterBuffer;
	return 0;
}