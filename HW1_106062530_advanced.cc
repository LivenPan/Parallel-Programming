#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include <algorithm>

#define	isOdd(x) ((x) & 1)
#define isEven(x) (!((x) & 1))

//Some compulsory variable.
int rankID = 0;					//Identify the process.
int numOfProcess = 0;			//Total number of process.
int dataOfperProcess = 0;		//data amount per process has.
int lastProcesscount = 0;
int generalProcesscount = 0;
float *receive_buff;
MPI_Comm customWorld = MPI_COMM_WORLD;
MPI_Group oldGroup, newGroup;
MPI_Offset offset;
MPI_Status status;

void mpi_custom_init(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
	MPI_Comm_rank(MPI_COMM_WORLD, &rankID);
}

void preserveLow(float *arrayOne, float *arrayTwo, int sendCount, int recvCount) {
	float *temp = (float*)malloc(sendCount * sizeof(float));
	int x = 0, y = 0;
	for (int i = 0; i < sendCount; i++) {
		if (y>=recvCount || (y < recvCount && x < sendCount && arrayOne[x] <= arrayTwo[y])) {
			temp[i] = arrayOne[x];
			x++;
		}
		else {
			temp[i] = arrayTwo[y];
			y++;
		}
	}	

	for (int i = 0; i<sendCount; i++)
		arrayOne[i] = temp[i];

	free(temp);
}

void preserveHigh(float *arrayOne, float *arrayTwo, int sendCount, int recvCount) {
	float *temp = (float*)malloc(sendCount * sizeof(float));
	int x = sendCount - 1, y = recvCount - 1;
	for (int i = sendCount - 1; i >= 0; i--) {
		if (y<0 || (y>=0 && x>=0 && arrayOne[x] >= arrayTwo[y])) {
			temp[i] = arrayOne[x];
			x--;
		}
		else {
			temp[i] = arrayTwo[y];
			y--;
		}
	}

	for (int i = 0; i<sendCount; i++)
		arrayOne[i] = temp[i];
		
	free(temp);
}

void exchangeWithright(float* localBuffer, int rankID, int sendCount, int recvCount) {

	//MPI_Send(localBuffer, dataOfperProcess, MPI_FLOAT, rankID - 1, 0, customWorld);

	MPI_Status status;

	//MPI_Recv(oddArray, dataOfperProcess, MPI_FLOAT, rankID - 1, 0, customWorld, &status);
	MPI_Sendrecv(localBuffer, sendCount, MPI_FLOAT, rankID + 1, 1, receive_buff, recvCount, MPI_FLOAT, rankID + 1, 0, customWorld, &status);

	preserveLow(localBuffer, receive_buff, sendCount, recvCount);
	
}

void exchangeWithleft(float* localBuffer, int rankID, int sendCount, int recvCount) {

	//MPI_Send(localBuffer, dataOfperProcess, MPI_FLOAT, rankID - 1, 0, customWorld);

	MPI_Status status;

	//MPI_Recv(evenArray, dataOfperProcess, MPI_FLOAT, rankID - 1, 0, customWorld, &status);

	MPI_Sendrecv(localBuffer, sendCount, MPI_FLOAT, rankID - 1, 0, receive_buff, recvCount, MPI_FLOAT, rankID - 1, 1, customWorld, &status);

	preserveHigh(localBuffer, receive_buff, sendCount, recvCount);

}

int main(int argc, char *argv[]) {

	//Check input legality.
	if (argc != 4) {
		printf("Please obey the following input format: Size_of_list, input_file_name, output_file_name.\n");
		return -1;
	}
	//MPI Init
	mpi_custom_init(argc, argv);


	//Handle numOfProcess > Size_of_listd
	if (atoi(argv[1]) < numOfProcess) {
		MPI_Comm_group(customWorld, &oldGroup);
		int ranges[][3] = { { atoi(argv[1]), numOfProcess - 1, 1 } };
		MPI_Group_range_excl(oldGroup, 1, ranges, &newGroup);
		MPI_Comm_create(customWorld, newGroup, &customWorld);
		if (customWorld == MPI_COMM_NULL) {
			MPI_Finalize();
			exit(0);
		}
		numOfProcess = atoi(argv[1]);
	}

	//Varaible setting
	/*float *localBuffer;	
	dataOfperProcess = (int)(ceil((double)atoi(argv[1])/(double)numOfProcess));
	generalProcesscount = dataOfperProcess;	
	lastProcesscount =  atoi(argv[1]) - rankID * dataOfperProcess; 
	offset = sizeof(float) * rankID * dataOfperProcess;

	//Handle remaining data
	if (rankID == numOfProcess - 1) {
		dataOfperProcess = lastProcesscount;
		//lastProcesscount = dataOfperProcess;
	}
	//lastProcesscount = dataOfperProcess + (atoi(argv[1]) % numOfProcess);
	localBuffer = (float*)malloc(dataOfperProcess * sizeof(MPI_FLOAT));
	receive_buff = (float*)malloc(lastProcesscount * sizeof(MPI_FLOAT));*/
	
	//// other approach
	dataOfperProcess     = (int)(ceil((double)atoi(argv[1])/(double)numOfProcess));
	generalProcesscount = dataOfperProcess;
	lastProcesscount = atoi(argv[1])- (dataOfperProcess * (numOfProcess-1));
	offset      = rankID * dataOfperProcess * sizeof(float);
	if(rankID == numOfProcess -1) dataOfperProcess = lastProcesscount;

	/// create memory space
	float *localBuffer;
	localBuffer = (float*)malloc(dataOfperProcess * sizeof(MPI_FLOAT));
	receive_buff = (float*)malloc(generalProcesscount * sizeof(MPI_FLOAT));
	

	//MPI Open file
	MPI_File fileInput, fileOutput;


	MPI_File_open(customWorld, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fileInput);

	//MPI Read file	
	MPI_File_read_at(fileInput, offset, localBuffer, dataOfperProcess, MPI_FLOAT,  MPI_STATUS_IGNORE);
	MPI_File_close(&fileInput);


	//in-process sort
	std::sort(localBuffer, localBuffer + dataOfperProcess);

	//between process

	for (int i = 0; i < numOfProcess; i++) {
		//odd-phase
		if (isOdd(i)) {
			if (isOdd(rankID)) {
				if (rankID == numOfProcess - 2) {
					exchangeWithright(localBuffer, rankID, dataOfperProcess, lastProcesscount);
				}
				else if (rankID != numOfProcess - 1) {
					exchangeWithright(localBuffer, rankID, dataOfperProcess, generalProcesscount);
				}
			}

			if (isEven(rankID)) {
				if (rankID == numOfProcess - 1) {
					exchangeWithleft(localBuffer, rankID, dataOfperProcess, generalProcesscount);
				}
				else if (rankID != 0) {
					exchangeWithleft(localBuffer, rankID, dataOfperProcess, dataOfperProcess);
				}
			}
		}

		//even-phase
		if (isEven(i)) {
			if (isOdd(rankID)) {
				if (rankID == numOfProcess - 1) {
					exchangeWithleft(localBuffer, rankID, dataOfperProcess, generalProcesscount);
				}
				else {
					exchangeWithleft(localBuffer, rankID, dataOfperProcess, dataOfperProcess);
				}
			}

			if (isEven(rankID)) {
				if (rankID == numOfProcess - 2) {
					exchangeWithright(localBuffer, rankID, dataOfperProcess, lastProcesscount);
				}
				else if (rankID != numOfProcess - 1) {
					exchangeWithright(localBuffer, rankID, dataOfperProcess, generalProcesscount);
				}
			}
		}		

	}


	MPI_File_open(customWorld, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fileOutput);

	MPI_File_write_at(fileOutput, offset, localBuffer, dataOfperProcess, MPI_FLOAT, MPI_STATUS_IGNORE);

	MPI_File_close(&fileOutput);

	free(localBuffer);
	free(receive_buff);
	
	MPI_Finalize();
	return 0;
}