#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mpi.h>

const int INF = 1000000000;
const int V = 20010;
int resultFinal[V][V] = { 0 };

__device__ __host__ int location(int i, int j, int N) {
	return i*N+j;
}

__device__ int judge(int a, int b, int c){
	return (a > b + c) ? b + c : a;
}

int* distanHost;
int* device_matrix;

__global__ void phase1(int* din, int blocking_factor, int N, int r, int MAX_BLCOK_DIM) {
	int bi, bj;
    
	bi = r;
	bj = r;
    
	extern __shared__ int DS[];

	int offset_i = blocking_factor * bi;
	int offset_j = blocking_factor * bj;
	int offset_r = blocking_factor * r;

	int i = threadIdx.y;
	int j = threadIdx.x;

	DS[location(i, j, blocking_factor)] = din[location(i+offset_i, j+offset_j, N)];
	DS[location(i+blocking_factor, j, blocking_factor)] = din[location(i+offset_i, j+offset_r, N)];
	DS[location(i+2*blocking_factor, j, blocking_factor)] = din[location(i+offset_r, j+offset_j, N)];
	__syncthreads();


	for (int k = 0; k < blocking_factor; k++) {
		if (DS[location(i, j, blocking_factor)] > DS[location(i+blocking_factor, k, blocking_factor)] + DS[location(k+2*blocking_factor, j, blocking_factor)]) {
            DS[location(i, j, blocking_factor)] = DS[location(i+blocking_factor, k, blocking_factor)] + DS[location(k+2*blocking_factor, j, blocking_factor)];
            DS[location(i+2*blocking_factor, j, blocking_factor)] = DS[location(i, j, blocking_factor)];
            DS[location(i+blocking_factor, j, blocking_factor)] = DS[location(i, j, blocking_factor)];
		}	

	}


	din[location(i+offset_i, j+offset_j, N)] = DS[location(i, j, blocking_factor)];
	__syncthreads();
}

__global__ void phase2(int* din, int blocking_factor, int N, int r, int MAX_BLCOK_DIM) {
	int bi, bj;
	
	if (blockIdx.x == 1) {
	
		bi = (r + blockIdx.y + 1) % (N/blocking_factor);
		bj = r;
	} else {
	
		bi = r;
		bj = (r + blockIdx.y + 1) % (N/blocking_factor);
            }

	extern __shared__ int DS[];
	
	int offset_i = blocking_factor * bi;
	int offset_j = blocking_factor * bj;
	int offset_r = blocking_factor * r;

	int i = threadIdx.y;
	int j = threadIdx.x;

	DS[location(i, j, blocking_factor)] = din[location(i+offset_i, j+offset_j, N)];
	DS[location(i+blocking_factor, j, blocking_factor)] = din[location(i+offset_i, j+offset_r, N)];
	DS[location(i+2*blocking_factor, j, blocking_factor)] = din[location(i+offset_r, j+offset_j, N)];
	__syncthreads();


	for (int k = 0; k < blocking_factor; k++) {
		if (DS[location(i, j, blocking_factor)] > DS[location(i+blocking_factor, k, blocking_factor)] + DS[location(k+2*blocking_factor, j, blocking_factor)]) {
            DS[location(i, j, blocking_factor)] = DS[location(i+blocking_factor, k, blocking_factor)] + DS[location(k+2*blocking_factor, j, blocking_factor)];
            if (r == bi) DS[location(i+2*blocking_factor, j, blocking_factor)] = DS[location(i, j, blocking_factor)];
            if (r == bj) DS[location(i+blocking_factor, j, blocking_factor)] = DS[location(i, j, blocking_factor)];
		}	

	}

	din[location(i+offset_i, j+offset_j, N)] = DS[location(i, j, blocking_factor)];
	__syncthreads();
}
__global__ void phase3(int* din, int blocking_factor, int N, int r, int MAX_BLCOK_DIM, int offset) {
	int bi, bj;
    
	bi = blockIdx.x + offset;
	bj = blockIdx.y;
     
	extern __shared__ int DS[];
	
	int offset_i = blocking_factor * bi;
	int offset_j = blocking_factor * bj;
	int offset_r = blocking_factor * r;

	int i = threadIdx.y;
	int j = threadIdx.x;


	DS[location(i, j, blocking_factor)] = din[location(i+offset_i, j+offset_j, N)];
	DS[location(i+blocking_factor, j, blocking_factor)] = din[location(i+offset_i, j+offset_r, N)];
	DS[location(i+2*blocking_factor, j, blocking_factor)] = din[location(i+offset_r, j+offset_j, N)];
	__syncthreads();
	
   
	for (int k = 0; k < blocking_factor; k++) {
		if (DS[location(i, j, blocking_factor)] > DS[location(i+blocking_factor, k, blocking_factor)] + DS[location(k+2*blocking_factor, j, blocking_factor)]) {
            DS[location(i, j, blocking_factor)] = DS[location(i+blocking_factor, k, blocking_factor)] + DS[location(k+2*blocking_factor, j, blocking_factor)];

		}	

	}
	

	din[location(i+offset_i, j+offset_j, N)] = DS[location(i, j, blocking_factor)];
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
	int blocking_factor = atoi(argv[3]);

	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	cudaSetDevice(rank);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, rank);
	printf("Dev%d. Name: %s\n", rank, prop.name);


	// read file
	FILE *fh_in;
	fh_in = fopen(INPUT_NAME, "r");
	int edgeNum, vertexNum;
	fscanf(fh_in, "%d %d", &vertexNum, &edgeNum);
	if (blocking_factor > vertexNum) blocking_factor = vertexNum;
	int vertexModify = vertexNum + (blocking_factor - ((vertexNum-1) % blocking_factor + 1));
 
	if(rank == 0) printf("Blocking factor: %d\n", blocking_factor);

	//allocate memory
	cudaMallocHost((void**) &distanHost, sizeof(int) * vertexModify*vertexModify);
	cudaMalloc((void**) &device_matrix, sizeof(int) * vertexModify*vertexModify);


	//initialize (store data in row major)
	for(int i = 0; i < vertexModify; i++){
		for(int j = 0; j < vertexModify; j++){
			if(i == j) distanHost[i * vertexModify + j] = 0;
			else distanHost[i * vertexModify + j] = INF;
		}
	}
	
	int a, b, weight;
	for(int i = 0; i < edgeNum; i++){
		fscanf(fh_in, "%d %d %d", &a, &b, &weight);
		distanHost[a * vertexModify + b] = weight;
	}

	fclose(fh_in);

	int MAX_BLCOK_DIM = blocking_factor > 32 ? 32 : blocking_factor;

	dim3 BLOCK_DIM(MAX_BLCOK_DIM, MAX_BLCOK_DIM);
	
	
	int blocks[3];

	dim3 grid_phase1(1);

	int round =  (vertexModify + blocking_factor - 1) / blocking_factor;
	blocks[1] = round;
	dim3 grid_phase2(2, blocks[1]-1);

	int num_blocks_per_thread = round / size;
	int row_offset = num_blocks_per_thread * rank * blocking_factor;
	if (rank == size - 1) num_blocks_per_thread += round % size;
	dim3 grid_phase3(num_blocks_per_thread, round);
	
	int cpy_idx = location(row_offset, 0, vertexModify);

	cudaMemcpy((void*) &(device_matrix[cpy_idx]), (void*) &(distanHost[cpy_idx]), sizeof(int) * vertexModify*blocking_factor*num_blocks_per_thread, cudaMemcpyHostToDevice);	

	for (int r = 0; r < round; r++) {

		int r_idx = location(r * blocking_factor, 0, vertexModify);
		
		if (r >= row_offset/blocking_factor && r < (row_offset/blocking_factor + num_blocks_per_thread)) {
            cudaMemcpy((void*) &(distanHost[r_idx]), (void*) &(device_matrix[r_idx]), sizeof(int) * vertexModify * blocking_factor, cudaMemcpyDeviceToHost);
            // MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
            MPI_Send(&distanHost[r_idx], vertexModify * blocking_factor, MPI_INT, (rank + 1) % 2, 0, MPI_COMM_WORLD);
        } else {
            // MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
            MPI_Recv(&distanHost[r_idx], vertexModify * blocking_factor, MPI_INT, (rank + 1) % 2, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
		

		cudaMemcpy((void*) &(device_matrix[r_idx]), (void*) &(distanHost[r_idx]), sizeof(int) * vertexModify * blocking_factor, cudaMemcpyHostToDevice);

		phase1<<< grid_phase1, BLOCK_DIM, sizeof(int)*3*blocking_factor*blocking_factor >>>(device_matrix, blocking_factor, vertexModify, r, MAX_BLCOK_DIM);
        cudaDeviceSynchronize();   
		phase2<<< grid_phase2, BLOCK_DIM, sizeof(int)*3*blocking_factor*blocking_factor >>>(device_matrix, blocking_factor, vertexModify, r, MAX_BLCOK_DIM);
        cudaDeviceSynchronize();  
		phase3<<< grid_phase3, BLOCK_DIM, sizeof(int)*3*blocking_factor*blocking_factor >>>(device_matrix, blocking_factor, vertexModify, r, MAX_BLCOK_DIM, row_offset/blocking_factor);
           
	}
	
	cudaMemcpy((void*) &(distanHost[cpy_idx]), (void*) &(device_matrix[cpy_idx]), sizeof(int) * vertexModify*blocking_factor*num_blocks_per_thread, cudaMemcpyDeviceToHost);
	
	
	if (rank == 0) {
		int send_idx = 0;
		int send_cnt = vertexModify*blocking_factor*num_blocks_per_thread;
		int recv_idx = location(num_blocks_per_thread * blocking_factor, 0, vertexModify);
		int recv_cnt = vertexModify*blocking_factor*(num_blocks_per_thread + round % size);
/*		
		int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                int dest, int sendtag,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int source, int recvtag,
                MPI_Comm comm, MPI_Status *status)
*/				
		MPI_Sendrecv(&distanHost[send_idx], send_cnt, MPI_INT, 1, 0, &distanHost[recv_idx], recv_cnt, MPI_INT, 1, MPI_ANY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } else {
		int recv_idx = 0;
		int recv_cnt = vertexModify*blocking_factor*(num_blocks_per_thread - round % size);
		int send_idx = location((num_blocks_per_thread - round % size) * blocking_factor, 0, vertexModify);
		int send_cnt = vertexModify*blocking_factor*num_blocks_per_thread;
		MPI_Sendrecv(&distanHost[send_idx], send_cnt, MPI_INT, 0, 0, &distanHost[recv_idx], recv_cnt, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }
	
	MPI_Barrier(MPI_COMM_WORLD);
	// output
 
 //Write output
	  int index = 0;
	  FILE *output = fopen(argv[2], "wb");
	  for (int i = 0; i < vertexNum; ++i) {
		  for (int j = 0; j < vertexNum; ++j) {
			  resultFinal[i][j] = distanHost[i * vertexModify + j];
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

	//free memory
	cudaFreeHost(distanHost);
	cudaFree(device_matrix);

	MPI_Barrier(MPI_COMM_WORLD);		
	MPI_Finalize();
	return 0;
}