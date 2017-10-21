#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdbool.h"

#define	isOdd(x) ((x) & 1)
#define isEven(x) (!((x) & 1))
#define swap(x,y) float t=x; x=y; y=t;

//Some compulsory variable.
int rankID = 0;					//Identify the process.
int numOfProcess = 0;			//Total number of process.
int dataOfperProcess = 0;  //data amount per process has
bool sorted = false;
MPI_Comm customWorld = MPI_COMM_WORLD;
MPI_Group oldGroup, newGroup;
MPI_Offset offset;


void mpi_custom_init(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
	MPI_Comm_rank(MPI_COMM_WORLD, &rankID);
}


void localSort(float* buffer, int index, int dataOfperProcess) {
	for (int i = index; i < dataOfperProcess - 1; i += 2) {
		if (buffer[i] > buffer[i + 1]) {
			swap(buffer[i], buffer[i + 1]);
			sorted = false;
		}
	}
}


int main(int argc, char *argv[]) {
	
	//Check input legality.
	if (argc != 4) {
		printf("Please obey the following input format: Size_of_list, input_file_name, output_file_name.\n");
		return -1;
	}

	//MPI Init
	mpi_custom_init(argc, argv);
	

	//Handle numOfProcess > Size_of_list
	if (atoi(argv[1]) < numOfProcess) {
		MPI_Comm_group(customWorld, &oldGroup);
		int ranges[][3] = { { atoi(argv[1]), numOfProcess - 1, 1 } };
		MPI_Group_range_excl(oldGroup, 1, ranges, &newGroup); //int MPI_Group_range_excl(MPI_Group group, int n, int ranges[][3],MPI_Group *newgroup)
		MPI_Comm_create(customWorld, newGroup, &customWorld);
		if (customWorld == MPI_COMM_NULL) {
			MPI_Finalize();
			exit(0);
		}
		numOfProcess = atoi(argv[1]);
	}

	//Varaible setting
	dataOfperProcess = atoi(argv[1]) / numOfProcess;
	int head = dataOfperProcess * rankID;
	int tail = head + dataOfperProcess - 1;
	offset = head * sizeof(MPI_FLOAT);

	//Handle remaining data
	if (rankID == numOfProcess - 1) {
		dataOfperProcess = dataOfperProcess + (atoi(argv[1]) % numOfProcess);
	}

	//Build localBuffer
	float *localBuffer;
	localBuffer = new float [dataOfperProcess];

	//MPI Open file
	MPI_File fileInput, fileOutput;

	MPI_File_open(customWorld, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fileInput);
	MPI_File_read_at(fileInput, offset, localBuffer, dataOfperProcess, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&fileInput);
	

	
	//Main process
	while (sorted != 1) {		
		sorted = true;
		//oddPhase
		if (rankID != (numOfProcess - 1)) {
			if (isOdd(tail)) {
				float tempBuffer;				
				MPI_Sendrecv(&localBuffer[dataOfperProcess - 1], 1, MPI_FLOAT, rankID + 1, 1, &tempBuffer, 1, MPI_FLOAT, rankID + 1, 0, customWorld, MPI_STATUS_IGNORE);
				
				if (localBuffer[dataOfperProcess - 1] > tempBuffer) {
					localBuffer[dataOfperProcess - 1] = tempBuffer;
					sorted = false;
				}
			}
		}		

		if (rankID != 0) {
			if (isEven(head)) {
				float tempBuffer2;
			
				MPI_Sendrecv(&localBuffer[0], 1, MPI_FLOAT, rankID - 1, 0, &tempBuffer2, 1, MPI_FLOAT, rankID - 1, 1, customWorld, MPI_STATUS_IGNORE);
			
				if (localBuffer[0] < tempBuffer2) {
					localBuffer[0] = tempBuffer2;
					sorted = false;
				}
			}
		}
		localSort(localBuffer, 1, dataOfperProcess);
		MPI_Barrier(customWorld);


		//evenPhase
		if (rankID != (numOfProcess - 1)) {
			if (isEven(tail)) {
				float tempBuffer;

			
				MPI_Sendrecv(&localBuffer[dataOfperProcess - 1], 1, MPI_FLOAT, rankID + 1, 1, &tempBuffer, 1, MPI_FLOAT, rankID + 1, 0, customWorld, MPI_STATUS_IGNORE);
			
				
				if (localBuffer[dataOfperProcess - 1] > tempBuffer) {
					localBuffer[dataOfperProcess - 1] = tempBuffer;
					sorted = false;
				}
			}
		}

		if (rankID != 0) {
			if (isOdd(head)) {
				float tempBuffer2;
				
				MPI_Sendrecv(&localBuffer[0], 1, MPI_FLOAT, rankID - 1, 0, &tempBuffer2, 1, MPI_FLOAT, rankID - 1, 1, customWorld, MPI_STATUS_IGNORE);
			
				if (localBuffer[0] < tempBuffer2) {
					localBuffer[0] = tempBuffer2;
					sorted = false;
				}
			}
		}
		localSort(localBuffer, 0, dataOfperProcess);

		
		bool temp = sorted;
		
		MPI_Allreduce(&temp, &sorted, 1, MPI_CHAR, MPI_BAND, customWorld);
		
		MPI_Barrier(customWorld);		
	
	}

	MPI_File_open(customWorld, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fileOutput);

	MPI_File_write_at(fileOutput, offset, localBuffer, dataOfperProcess, MPI_FLOAT, MPI_STATUS_IGNORE);

	MPI_File_close(&fileOutput);

	
	delete(localBuffer);

	MPI_Barrier(customWorld);

	MPI_Finalize();
}