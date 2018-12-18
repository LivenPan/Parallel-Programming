#include <mpi.h>
#include <limits>
#include <stdlib.h>
#include <stdio.h>

#define MAX 1500000

int rankID, MPIsize;
int numOfprocess;
int vertices, edges;
int *neighbor, *result, *adjancy, *needUpdated, *output;

void all_pair_shortest_path() {

	int localDone = false, allDone = false;
	while (!allDone) {
		localDone = true;
		for (int j = 0; j<vertices; ++j) {
			if (j == rankID) continue;
			if (adjancy[j] != MAX) {
				for (int i = 0; i<vertices; ++i) {
					needUpdated[i] = result[i] + adjancy[j];
				}
				MPI_Sendrecv(needUpdated, vertices, MPI_INT, j, 0, neighbor, vertices, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for (int i = 0; i < vertices; ++i) {
					if (neighbor[i] < result[i]) {
						result[i] = neighbor[i];
						localDone = false;
					}
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&localDone, &allDone, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
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

	adjancy = new int[vertices];
	result = new int[vertices];
	needUpdated = new int[vertices];
	neighbor = new int[vertices];

	for (int i = 0; i<vertices; ++i) {
		adjancy[i] = MAX;
		result[i] = MAX;
		needUpdated[i] = MAX;
		neighbor[i] = MAX;
	}
	result[rankID] = 0;
	needUpdated[rankID] = 0;
	int from, to, weight;
	for (int i = 0; i<edges; ++i) {
		fscanf(fp, "%d %d %d\n", &from, &to, &weight);
		if (rankID == from)
			adjancy[to] = weight;
		if (rankID == to)
			adjancy[from] = weight;
	}
	fclose(fp);

	all_pair_shortest_path();

	output = new int[vertices*vertices];
	MPI_Gather(result, vertices, MPI_INT, output, vertices, MPI_INT, 0, MPI_COMM_WORLD);

	if (rankID == 0) {
		//Write output
		FILE* fpo = fopen(argv[2], "w");
		if (!fpo) {
			printf("Open file failed");
			return -1;
		}
		for (int i = 0; i < vertices; i++) {
			for (int j = 0; j < vertices; j++) {
				fprintf(fpo, "%d ", output[i * vertices + j]);
				if (j == vertices - 1)
					fprintf(fpo, "\n");
			}
		}
	}

	MPI_Finalize();

	return 0;
}