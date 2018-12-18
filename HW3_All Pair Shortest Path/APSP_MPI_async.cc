#include <mpi.h>
#include <limits>
#include <stdlib.h>
#include <stdio.h>

#define CALC 0
#define TOKEN 1
#define TERMINATE 2
#define black 0
#define white 1
#define MAX 1500000

int vertices, edges;
int *result;
int rankID, MPIsize;
int numOfprocess;
MPI_Status status;

int *allSource, *localDIST, *receiver, *localAdjancy, *token;

void all_pair_shortest_path() {
	for (int i = 0; i < vertices; ++i) {
		if (i == rankID) continue;
		else if (localAdjancy[i] > 0) {
			if (i < rankID) token[0] = black;
			for (int j = 0; j < vertices; ++j)
				localDIST[j] = allSource[j] + localAdjancy[i];
			MPI_Send(localDIST, vertices, MPI_INT, i, CALC, MPI_COMM_WORLD);
		}
	}
	if (rankID == 0) {
		MPI_Send(token, vertices, MPI_INT, 1, TOKEN, MPI_COMM_WORLD);
		token[0] = white;
	}

	bool change = false;
	while (MPI_Recv(receiver, vertices, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status) == MPI_SUCCESS) {
		if (status.MPI_TAG == CALC) {
			for (int i = 0; i < vertices; ++i) {
				if (allSource[i] > receiver[i]) {
					allSource[i] = receiver[i];
					change = true;
				}
			}
		}
		else if (status.MPI_TAG == TOKEN) {
			if (change) {
				for (int i = 0; i < vertices; ++i) {
					if (i == rankID) continue;
					else if (localAdjancy[i] > 0) {
						if (i < rankID) token[0] = black;
						for (int j = 0; j < vertices; ++j)
							localDIST[j] = allSource[j] + localAdjancy[i];
						MPI_Send(localDIST, vertices, MPI_INT, i, CALC, MPI_COMM_WORLD);
						change = false;
					}
				}
			}
			if (rankID == 0) {
				if (receiver[0] == white) {
					MPI_Send(token, vertices, MPI_INT, 1, TERMINATE, MPI_COMM_WORLD);
					break;
				}
				else {
					token[0] = white;
					MPI_Send(token, vertices, MPI_INT, 1, TOKEN, MPI_COMM_WORLD);
				}
			}
			else {
				token[0] = token[0] & receiver[0];
				MPI_Send(token, vertices, MPI_INT, (rankID + 1) % MPIsize, TOKEN, MPI_COMM_WORLD);
				token[0] = white;
			}
		}
		else if (status.MPI_TAG == TERMINATE) {
			if (rankID != vertices - 1)
				MPI_Send(token, vertices, MPI_INT, (rankID + 1) % MPIsize, TERMINATE, MPI_COMM_WORLD);
			break;
		}
	}
}

int main(int argc, char *argv[]) {

	//Check input legality.
	if (argc != 4) {
		printf("Please obey the following input format: input_file_name, output_file_name, number of threads or MPI processes.\n");
		return -1;
	}

	numOfprocess = atoi(argv[3]);

	//Set MPI Env.
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &rankID);
	

	//Read input
	FILE* fp = fopen(argv[1], "r");

	if (!fp) {
		printf("Open file failed");
		return -1;
	}
	fscanf(fp, "%d%d", &vertices, &edges);

	
	allSource = new int[vertices];				
	localDIST = new int[vertices];
	receiver = new int[vertices];			
	localAdjancy = new int[vertices];
	token = new int[vertices];					

	if (rankID == 0)
		result = new int[vertices*vertices];

	for (int i = 0; i < vertices; ++i) {
		allSource[i] = MAX;
		localDIST[i] = MAX;		
		localAdjancy[i] = -1;
	}
	allSource[rankID] = 0;
	localDIST[rankID] = 0;
	token[0] = white;

	int from, to, weight;
	for (int i = 0; i < edges; i++) {
		fscanf(fp, "%d %d %d", &from, &to, &weight);
		if (rankID == from) 
			localAdjancy[to] = weight;
		if (rankID == to) 
			localAdjancy[from] = weight;
	}
	fclose(fp);

	all_pair_shortest_path();

	MPI_Gather(allSource, vertices, MPI_INT, result, vertices, MPI_INT, 0, MPI_COMM_WORLD);

	if (rankID == 0) {
		FILE* fpo = fopen(argv[2], "w");
		for (int i = 0; i < vertices; i++) {
			for (int j = 0; j < vertices; j++) {
				fprintf(fpo, "%d ", result[i * vertices + j]);
			}
			fprintf(fpo, "\n");
		}
		fclose(fpo);
		
	}
	
	MPI_Finalize();
	return 0;
}