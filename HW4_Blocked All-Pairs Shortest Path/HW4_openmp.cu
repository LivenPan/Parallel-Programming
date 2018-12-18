#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

const int INF = 1000000000;
const int V = 20010;
int resultFinal[V][V] = { 0 };


__device__ __host__ int location(int i, int j, int N) {
	return i*N+j;
}


__device__ int judge(int a, int b, int c){
	return (a > b + c) ? b + c : a;
}

int* distanhost;
int** device_matrix;

__global__ void phase1(int* din, int blockFactor, int N, int r, int MAX_BLCOK_DIM) {
	int bi, bj;
    
	bi = r;
	bj = r;
    
	extern __shared__ int DS[];

	int offset_i = blockFactor * bi;
	int offset_j = blockFactor * bj;
	int offset_r = blockFactor * r;

	int i = threadIdx.y;
	int j = threadIdx.x;

	DS[location(i+blockFactor, j, blockFactor)] = din[location(i+offset_i, j+offset_r, N)];
	DS[location(i+2*blockFactor, j, blockFactor)] = din[location(i+offset_r, j+offset_j, N)];
	__syncthreads();

	for (int k = 0; k < blockFactor; k++) {
		if (DS[location(i, j, blockFactor)] > DS[location(i+blockFactor, k, blockFactor)] + DS[location(k+2*blockFactor, j, blockFactor)]) {
            DS[location(i, j, blockFactor)] = DS[location(i+blockFactor, k, blockFactor)] + DS[location(k+2*blockFactor, j, blockFactor)];
            DS[location(i+2*blockFactor, j, blockFactor)] = DS[location(i, j, blockFactor)];
            DS[location(i+blockFactor, j, blockFactor)] = DS[location(i, j, blockFactor)];
		}	

	}


	din[location(i+offset_i, j+offset_j, N)] = DS[location(i, j, blockFactor)];
	__syncthreads();
}

__global__ void phase2(int* din, int blockFactor, int N, int r, int MAX_BLCOK_DIM) {
	int bi, bj;
	
	if (blockIdx.x == 1) {
	
		bi = (r + blockIdx.y + 1) % (N/blockFactor);
		bj = r;
	} else {
	
		bi = r;
		bj = (r + blockIdx.y + 1) % (N/blockFactor);
            }

	extern __shared__ int DS[];
	
	int offset_i = blockFactor * bi;
	int offset_j = blockFactor * bj;
	int offset_r = blockFactor * r;

	int i = threadIdx.y;
	int j = threadIdx.x;

	DS[location(i, j, blockFactor)] = din[location(i+offset_i, j+offset_j, N)];
	DS[location(i+blockFactor, j, blockFactor)] = din[location(i+offset_i, j+offset_r, N)];
	DS[location(i+2*blockFactor, j, blockFactor)] = din[location(i+offset_r, j+offset_j, N)];
	__syncthreads();


	for (int k = 0; k < blockFactor; k++) {
		if (DS[location(i, j, blockFactor)] > DS[location(i+blockFactor, k, blockFactor)] + DS[location(k+2*blockFactor, j, blockFactor)]) {
            DS[location(i, j, blockFactor)] = DS[location(i+blockFactor, k, blockFactor)] + DS[location(k+2*blockFactor, j, blockFactor)];
            if (r == bi) DS[location(i+2*blockFactor, j, blockFactor)] = DS[location(i, j, blockFactor)];
            if (r == bj) DS[location(i+blockFactor, j, blockFactor)] = DS[location(i, j, blockFactor)];
		}	

	}

	din[location(i+offset_i, j+offset_j, N)] = DS[location(i, j, blockFactor)];
	__syncthreads();
}
__global__ void phase3(int* din, int blockFactor, int N, int r, int MAX_BLCOK_DIM, int offset) {
	int bi, bj;
    
	bi = blockIdx.x + offset;
	bj = blockIdx.y;
     
	extern __shared__ int DS[];
	
	int offset_i = blockFactor * bi;
	int offset_j = blockFactor * bj;
	int offset_r = blockFactor * r;

	int i = threadIdx.y;
	int j = threadIdx.x;

	DS[location(i, j, blockFactor)] = din[location(i+offset_i, j+offset_j, N)];
	DS[location(i+blockFactor, j, blockFactor)] = din[location(i+offset_i, j+offset_r, N)];
	DS[location(i+2*blockFactor, j, blockFactor)] = din[location(i+offset_r, j+offset_j, N)];
	__syncthreads();
	

	for (int k = 0; k < blockFactor; k++) {
		if (DS[location(i, j, blockFactor)] > DS[location(i+blockFactor, k, blockFactor)] + DS[location(k+2*blockFactor, j, blockFactor)]) {
            DS[location(i, j, blockFactor)] = DS[location(i+blockFactor, k, blockFactor)] + DS[location(k+2*blockFactor, j, blockFactor)];
		}	
	}
	
	din[location(i+offset_i, j+offset_j, N)] = DS[location(i, j, blockFactor)];
	__syncthreads();
}

int main(int argc, char* argv[]) {

	//Check input legality.
	if (argc != 4) {
		printf("Please obey the following input format: ./%s input_file_name output_file_name BlockSize.\n", argv[0]);
		return -1;
	}
	
	//Read input
	const char *INPUT_NAME = argv[1];
	
	int blockFactor = atoi(argv[3]);

	int num_devices = 1;
	cudaGetDeviceCount(&num_devices);
	#pragma omp parallel num_threads(num_devices)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, omp_get_thread_num());		
	
	}

  
	// read file
	FILE *fh_in;
	fh_in = fopen(INPUT_NAME, "r");
	int edgeNum, vertexNum;
	fscanf(fh_in, "%d %d", &vertexNum, &edgeNum);
	if (blockFactor > vertexNum) blockFactor = vertexNum;
	int VERTEX_EXT = vertexNum + (blockFactor - ((vertexNum-1) % blockFactor + 1));
 
	printf("Blocking factor: %d\n", blockFactor);

	//Allocate
	cudaMallocHost((void**) &distanhost, sizeof(int) * VERTEX_EXT*VERTEX_EXT);
	device_matrix = (int**) malloc(sizeof(int*) * num_devices);
	#pragma omp parallel num_threads(num_devices)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaMalloc((void**) &device_matrix[omp_get_thread_num()], sizeof(int) * VERTEX_EXT*VERTEX_EXT);
	}

	
	for(int i = 0; i < VERTEX_EXT; ++i){
		for(int j = 0; j < VERTEX_EXT; ++j){
			if(i == j) distanhost[i * VERTEX_EXT + j] = 0;
			else distanhost[i * VERTEX_EXT + j] = INF;
		}
	}

	int a, b, weight;
	for(int i = 0; i < edgeNum; i++){
		fscanf(fh_in, "%d %d %d", &a, &b, &weight);	
		distanhost[a * VERTEX_EXT + b] = weight;
	}

	fclose(fh_in);
	int MAX_BLCOK_DIM = blockFactor > 32 ? 32 : blockFactor;


	dim3 BLOCK_DIM(MAX_BLCOK_DIM, MAX_BLCOK_DIM);
	
	
	int blocks[3];	
	dim3 grid_phase1(1);
	int round =  (VERTEX_EXT + blockFactor - 1) / blockFactor;
	blocks[1] = round;
	dim3 grid_phase2(2, blocks[1]-1);


	#pragma omp parallel num_threads(num_devices)
	{
		int t_id = omp_get_thread_num();
		cudaSetDevice(t_id);

		int num_blocks_per_thread = round / num_devices;
		int row_offset = num_blocks_per_thread * t_id * blockFactor;
		//allocate the remain num
		if (t_id == num_devices-1)
			num_blocks_per_thread += round % num_devices;

		dim3 grid_phase3(num_blocks_per_thread, round);
		
		
			
		
		int cpy_idx = location(row_offset, 0, VERTEX_EXT);

		cudaMemcpy((void*) &(device_matrix[t_id][cpy_idx]), (void*) &(distanhost[cpy_idx]), sizeof(int) * VERTEX_EXT*blockFactor*num_blocks_per_thread, cudaMemcpyHostToDevice);

		for (int r = 0; r < round; r++) {            
			int r_idx = location(r * blockFactor, 0, VERTEX_EXT);
		
			if (r >= row_offset/blockFactor && r < (row_offset/blockFactor + num_blocks_per_thread)) {

				cudaMemcpy((void*) &(distanhost[r_idx]), (void*) &(device_matrix[t_id][r_idx]), sizeof(int) * VERTEX_EXT * blockFactor, cudaMemcpyDeviceToHost);
			}
			#pragma omp barrier

			cudaMemcpy((void*) &(device_matrix[t_id][r_idx]), (void*) &(distanhost[r_idx]), sizeof(int) * VERTEX_EXT * blockFactor, cudaMemcpyHostToDevice);

			phase1<<< grid_phase1, BLOCK_DIM, sizeof(int)*3*blockFactor*blockFactor >>>(device_matrix[t_id], blockFactor, VERTEX_EXT, r, MAX_BLCOK_DIM);
			cudaDeviceSynchronize();
			phase2<<< grid_phase2, BLOCK_DIM, sizeof(int)*3*blockFactor*blockFactor >>>(device_matrix[t_id], blockFactor, VERTEX_EXT, r, MAX_BLCOK_DIM);
			cudaDeviceSynchronize();
			phase3<<< grid_phase3, BLOCK_DIM, sizeof(int)*3*blockFactor*blockFactor >>>(device_matrix[t_id], blockFactor, VERTEX_EXT, r, MAX_BLCOK_DIM, row_offset/blockFactor);           
		}
		cudaMemcpy((void*) &(distanhost[cpy_idx]), (void*) &(device_matrix[t_id][cpy_idx]), sizeof(int) * VERTEX_EXT*blockFactor*num_blocks_per_thread, cudaMemcpyDeviceToHost);
		#pragma omp barrier
	}
	

	//Write output
	int index = 0;
	FILE *output = fopen(argv[2], "wb");
	for (int i = 0; i < vertexNum; ++i) {
		for (int j = 0; j < vertexNum; ++j) {
			resultFinal[i][j] = distanhost[i * VERTEX_EXT + j];
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


	//Free
	cudaFreeHost(distanhost);
	#pragma omp parallel num_threads(num_devices)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaFree(device_matrix[omp_get_thread_num()]);
	}

	return 0;
}