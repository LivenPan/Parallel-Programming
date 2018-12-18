#include <iostream>
#include <limits>
#include <pthread.h>
#include <queue>
#include <stdlib.h>
#include <stdio.h>

#define MAX	2147483647
int numOfthreads;
int vertices, edges;
double adjancy[2010 * 2010];	
double A[2010][2010];

struct need {
	int threads;
	int vertices;
	double* dist;
	double* adjancy;
};

struct thread {
	int id;
	need* nd;
};

double* dijkstra(int vertices, double* adjancy, int source) {
	std::priority_queue<std::pair<double, int> > pq;
	double* dijDistance = new double[vertices];
	bool visited[vertices];

	for (int i = 0; i < vertices; i++) {
		dijDistance[i] = std::numeric_limits<double>::infinity();
		visited[i] = false;
	}
	dijDistance[source] = 0;
	pq.push(std::make_pair(0, source));

	while (!pq.empty()) {
		std::pair<double, int> curr = pq.top();
		int vertex = curr.second;
		pq.pop();

		if (!visited[vertex]) {
			visited[vertex] = true;
			for (int i = 0; i < vertices; i++) {
				if (!visited[i] && dijDistance[vertex] + adjancy[vertex * vertices + i] < dijDistance[i]) {
					dijDistance[i] = dijDistance[vertex] + adjancy[vertex * vertices + i];
					pq.push(std::make_pair(-dijDistance[i], i));
				}
			}
		}
	}

	return dijDistance;
}

void* secondPhase(void * ptr) {
	thread* tdata = (thread*)ptr;
	need* nd = tdata->nd;

	int id = tdata->id;
	int vertices = nd->vertices;
	int threads = nd->threads;
	double* adjancy = nd->adjancy;
	double* dist = nd->dist;

	for (int i = id; i < vertices; i += threads) {
		double* tmp = dijkstra(vertices, adjancy, i);
		for (int j = 0; j < vertices; j++) {
			dist[i * vertices + j] = tmp[j];
		}
	}
	return 0;
}

double* mainFunction(int vertices, double* adjancy, int threads) {
	double* dist = new double[vertices * vertices];
	pthread_t th[threads];
	need nd;
	nd.threads = threads;
	nd.dist = dist;
	nd.vertices = vertices;
	nd.adjancy = adjancy;
	thread threadData[threads];
	for (int i = 0; i < threads; i++) {
		threadData[i].id = i;
		threadData[i].nd = &nd;
		int status = pthread_create(&th[i], NULL, secondPhase, (void*)&threadData[i]);
    if(status){
      printf("Failed to creat thread");
      return NULL;
    }
		
	}
	for (int i = 0; i < threads; i++) {
		pthread_join(th[i], NULL);
	}
	return dist;
}

int main(int argc, char* argv[]) {

	//Check input legality.
	if (argc != 4) {
		printf("Please obey the following input format: input_file_name, output_file_name, number of threads or MPI processes.\n");
		return -1;
	}

	numOfthreads = atoi(argv[3]);

	//Read input
	FILE* fp = fopen(argv[1], "r");

	if (!fp) {
		printf("Open file failed");
		return -1;
	}
	fscanf(fp, "%d%d", &vertices, &edges);	

	//initialize temp matrix
	for (int i = 0; i < vertices; i++)
		for (int j = 0; j < vertices; j++)
			A[i][j] = MAX;

	//Enter input
	int v1, v2, weight;
	for (int i = 0; i < edges; i++) {
		fscanf(fp, "%d %d %d", &v1, &v2, &weight);
		A[v1][v2] = weight;
		A[v2][v1] = weight;
	}

	fclose(fp);

	//2D array -> 1D array
	for (int i = 0, index = 0; i < vertices; i++) {
		for (int j = 0; j < vertices; j++) {
			adjancy[index] = A[i][j];
			index++;
		}
	}
	double* result = mainFunction(vertices, adjancy, numOfthreads);

	//Write output
	FILE* fpo = fopen(argv[2], "w");
	if (!fpo) {
		printf("Open file failed");
		return -1;
	}

	for (int i = 0; i < vertices; i++) {
		for (int j = 0; j < vertices; j++) {
			fprintf(fpo, "%d ", (int)result[i * vertices + j]);
			if (j == vertices - 1)
				fprintf(fpo, "\n");
		}
	}
	fclose(fpo);

	return 0;
}