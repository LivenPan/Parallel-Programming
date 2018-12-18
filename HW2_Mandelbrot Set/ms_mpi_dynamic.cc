#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <assert.h>
#include <iostream>

#define terminateTag 0
#define dataTag 1
#define resultTag 2

//double start,end;
double left = 0, right = 0;	        //real-axis min & max
double lower = 0, upper = 0;		//image-axis min &max
int num_threads = 0, width = 0, height = 0;
const char* filename;
int rankID = 0;						//Identify the process.
int numOfProcess = 0;				//Total number of process.
int *masterBuffer;
int *result;

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

void Master() {

	// Init variables for master 
	MPI_Status status;
	int totalRow = 0;
	
	// Send row number to all slave
	for (int i = 1; i < numOfProcess; i++) {
		MPI_Send(&totalRow, 1, MPI_INT, i, dataTag, MPI_COMM_WORLD);
		totalRow++;
	}

	int finishRow = 0;
	// If slaves idle, keep sending work
	while (finishRow < height) {

		//Receive finished row
		MPI_Recv(masterBuffer, width + 1, MPI_INT, MPI_ANY_SOURCE, resultTag, MPI_COMM_WORLD, &status);

		int doneSlave = status.MPI_SOURCE;
		
		int receivedRow = masterBuffer[0];
		
		for (int i = 0; i < width; i++) {
			result[masterBuffer[0] * width + i] = masterBuffer[i+1];
		}

		// increment row to send the next one
		finishRow++;
   
		if (totalRow < height) {
			// Send data to slave which just finished.
			MPI_Send(&totalRow, 1, MPI_INT, doneSlave, dataTag, MPI_COMM_WORLD);
			totalRow++;
		}
		else {
			MPI_Send(0, 0, MPI_INT, doneSlave, terminateTag, MPI_COMM_WORLD);
		}  
	}		
}

int judge(double tempX, double tempY) {
	int repeats = 0;
	double x = 0;
	double y = 0;
	double length_squared = 0.0;
	while (repeats < 100000 && length_squared < 4.0) {
		double temp = x * x - y * y + tempX;
		y = 2 * x * y + tempY;
		x = temp;
		length_squared = x * x + y * y;
		++repeats;
	}
	return repeats;
}

void Slave() {

	//double total = 0;
  // Init Slave variables
	double x0 = (right - left) / width;
	double y0 = (upper - lower) / height;

	// Temporary variables
	double tempX = 0, tempY = 0;
	int *Slave;
	Slave = new int[width + 1];
	int row = 0;
	MPI_Status status;
	

	// Receive row which needed to be calculated
	MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	while (status.MPI_TAG == dataTag) {
    
    
		// Check for terminate tag
		if (status.MPI_TAG == terminateTag) {
			exit(0);
		}
   	//start = MPI_Wtime();
		Slave[0] = row;

		// Row calculation (only one calculation since it's a row)
		tempY = lower + row * y0;
	

		for (int x = 0; x < width; x++) {
			// Column calculation
			tempX = left + x * x0;
			Slave[x + 1] = judge(tempX, tempY);		   	   
		}
		//end = MPI_Wtime();	
		//total = total + (end-start);
		// A row is computed, send that row back to master
		MPI_Send(Slave, width + 1, MPI_INT, 0, resultTag, MPI_COMM_WORLD);
		MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		
	}
	//std::cout<<"rank id :" <<rankID << "Computation time :" << total << std::endl;
	delete[] Slave;
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
	//double s = MPI_Wtime();	
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
	MPI_Comm_rank(MPI_COMM_WORLD, &rankID);

	masterBuffer = new int[width + 1];
	result = new int[width * height];
	
	if(numOfProcess == 1){
		/* allocate memory for image */
	    int* image = (int*)malloc(width * height * sizeof(int));
	    assert(image);

	    /* mandelbrot set */
	    for (int j = 0; j < height; ++j) {
	        double y0 = j * ((upper - lower) / height) + lower;
	        for (int i = 0; i < width; ++i) {
	            double x0 = i * ((right - left) / width) + left;

	            int repeats = 0;
	            double x = 0;
	            double y = 0;
	            double length_squared = 0;
	            while (repeats < 100000 && length_squared < 4) {
	                double temp = x * x - y * y + x0;
	                y = 2 * x * y + y0;
	                x = temp;
	                length_squared = x * x + y * y;
	                ++repeats;
	            }
	            image[j * width + i] = repeats;
	        }
	    }

	    /* draw and cleanup */
	    write_png(filename, width, height, image);
	    free(image);
		return 0;
	}
	else{

		if (rankID == 0) {
			Master();
		}

		else {   
			Slave();
		}

		if (rankID == 0) {
			write_png(filename, width, height, result);
		}
		
		//if (rankID == 0) {
		//	std::cout << MPI_Wtime() - s << std::endl;
		//}
		delete[] result;
		delete[] masterBuffer;
		
		MPI_Finalize();	
	}
	return 0;
}