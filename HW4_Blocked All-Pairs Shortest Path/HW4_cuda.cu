#include <iostream>

using namespace std;

const int INF = 1000000000;
const int V = 20010;
int resultFinal[V][V] = { 0 };
int *distanHost;

static __global__ void phase1(int rounds, int vertexNum, int* dist, int blockSize) {
	extern __shared__ int shared_dist[];

	int x = threadIdx.x,
      y = threadIdx.y,
		  i = x + rounds * blockSize,
		  j = y + rounds * blockSize;

	shared_dist[x * blockSize + y] = (i < vertexNum && j < vertexNum) ? dist[i * vertexNum + j] : INF;
	__syncthreads();

#pragma unroll
	for (int k = 0; k < blockSize; ++k) {
		int tmp = shared_dist[x * blockSize + k] + shared_dist[k * blockSize + y];
		if (tmp < shared_dist[x * blockSize + y]) shared_dist[x * blockSize + y] = tmp;
		__syncthreads();
	}
	if (i < vertexNum && j < vertexNum) 
    dist[i * vertexNum + j] = shared_dist[x * blockSize + y];
	__syncthreads();
}

static __global__ void phase2(int rounds, int vertexNum, int* dist, int blockSize) {
	if (blockIdx.x == rounds) return;

	extern __shared__ int shared_mem[];
	int* shared_pivot = &shared_mem[0];
	int* shared_dist = &shared_mem[blockSize * blockSize];

	int x = threadIdx.x,
		y = threadIdx.y,
		i = x + rounds * blockSize,
		j = y + rounds * blockSize;

	shared_pivot[x * blockSize + y] = (i < vertexNum && j < vertexNum) ? dist[i * vertexNum + j] : INF;

	if (blockIdx.y == 0)
		j = y + blockIdx.x * blockSize;
	else
		i = x + blockIdx.x * blockSize;

	if (i >= vertexNum || j >= vertexNum) return;
	shared_dist[x * blockSize + y] = (i < vertexNum && j < vertexNum) ? dist[i * vertexNum + j] : INF;
	__syncthreads();

	if (blockIdx.y == 1) {
#pragma unroll
		for (int k = 0; k < blockSize; ++k) {
			int tmp = shared_dist[x * blockSize + k] + shared_pivot[k * blockSize + y];
			if (tmp < shared_dist[x * blockSize + y]) shared_dist[x * blockSize + y] = tmp;
		}
	}
	else {
#pragma unroll
		for (int k = 0; k < blockSize; ++k) {
			int tmp = shared_pivot[x * blockSize + k] + shared_dist[k * blockSize + y];
			if (tmp < shared_dist[x * blockSize + y]) shared_dist[x * blockSize + y] = tmp;
		}
	}

	if (i < vertexNum && j < vertexNum) dist[i * vertexNum + j] = shared_dist[x * blockSize + y];
}

static __global__ void phase3(int rounds, int vertexNum, int* dist, int blockSize) {
	if (blockIdx.x == rounds || blockIdx.y == rounds) return;

	extern __shared__ int shared_mem[];
	int* shared_pivot_row = &shared_mem[0];
	int* shared_pivot_col = &shared_mem[blockSize * blockSize];

	int x = threadIdx.x,
		y = threadIdx.y,
		i = x + blockIdx.x * blockDim.x,
		j = y + blockIdx.y * blockDim.y,
		i_col = y + rounds * blockSize,
		j_row = x + rounds * blockSize;

	shared_pivot_row[x * blockSize + y] = (i < vertexNum && i_col < vertexNum) ? dist[i * vertexNum + i_col] : INF;
	shared_pivot_col[x * blockSize + y] = (j < vertexNum && j_row < vertexNum) ? dist[j_row * vertexNum + j] : INF;
	__syncthreads();

	if (i >= vertexNum || j >= vertexNum) return;
	int dij = dist[i * vertexNum + j];

#pragma unroll
	for (int k = 0; k < blockSize; ++k) {
		int tmp = shared_pivot_row[x * blockSize + k] + shared_pivot_col[k * blockSize + y];
		if (tmp < dij) dij = tmp;
	}
	dist[i * vertexNum + j] = dij;
}

int main(int argc, char* argv[]) {

	//Check input legality.
	if (argc != 4) {
		printf("Please obey the following input format: ./%s input_file_name output_file_name BlockSize.\n", argv[0]);
		return -1;
	}

	//Read input
	FILE *input;
	int vertexNum = 0, edgeNum = 0;
	input = fopen(argv[1], "r");
	fscanf(input, "%d %d", &vertexNum, &edgeNum);
	distanHost = new int[V * V]();

	//Initialize matrix
	for (int i = 0; i < vertexNum; i++)
		for (int j = 0; j < vertexNum; j++)
			distanHost[i * vertexNum + j] = (i == j) ? 0 : INF;

	//Construct matrix
	while (--edgeNum >= 0) {
		int from, to, weight;
		fscanf(input, "%d %d %d", &from, &to, &weight);
		distanHost[from * vertexNum + to] = weight;
	}
	fclose(input);

	int *distanDev;	
	int blockSize = atoi(argv[3]);
	int rounds = (vertexNum + blockSize - 1) / blockSize;
	ssize_t  sz = sizeof(int) * vertexNum * vertexNum;
	cudaSetDevice(1);
	cudaMalloc(&distanDev, sz);
	cudaMemcpy(distanDev, distanHost, sz, cudaMemcpyHostToDevice);
	

	//Compute B กั B pivot block
	dim3 grid1(1, 1);
	dim3 grid2(rounds, 2);
	dim3 grid3(rounds, rounds);
	dim3 block(blockSize, blockSize);

	for (int r = 0; r < rounds; ++r) {
		phase1 <<< grid1, block, blockSize * blockSize * sizeof(int) >>> (r, vertexNum, distanDev, blockSize);

		phase2 <<< grid2, block, blockSize * blockSize * sizeof(int) * 2 >>> (r, vertexNum, distanDev, blockSize);

		phase3 <<< grid3, block, blockSize * blockSize * sizeof(int) * 2 >>> (r, vertexNum, distanDev, blockSize);
	}

	cudaMemcpy(distanHost, distanDev, sz, cudaMemcpyDeviceToHost);
	cudaFree(distanDev);	

	//Write output
	int index = 0;
	FILE *output = fopen(argv[2], "wb");
	for (int i = 0; i < vertexNum; ++i) {
		for (int j = 0; j < vertexNum; ++j) {
			resultFinal[i][j] = distanHost[index];
			index++;
		}
	}

	for (int i = 0; i < vertexNum; ++i) {
		for (int j = 0; j < vertexNum; ++j) {
			if (resultFinal[i][j] >= INF)
				resultFinal[i][j] = INF;
		}
		fwrite(resultFinal[i], sizeof(int), vertexNum, output);
	}

	fclose(output);
	return 0;
}