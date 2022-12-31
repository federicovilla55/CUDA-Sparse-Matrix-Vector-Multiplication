#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CHECK(call) \
{ \
  const cudaError_t err = call; \
  if (err != cudaSuccess) { \
    printf("%s in %s at line %d\n", cudaGetErrorString(err), \
    __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
}

#define CHECK_KERNELCALL() \
{ \
  const cudaError_t err = cudaGetLastError(); \
  if (err != cudaSuccess) { \
    printf("%s in %s at line %d\n", cudaGetErrorString(err), \
    __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
}

// Naive CUDA implementation of SPMV using CSR
__global__ void spmv_csr(const int num_rows, const int * row_ptr, const int * col_ind, 
                         const float * values, const float * x , float * y){

    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if( row < num_rows ){
        float dotProduct = 0;

        int row_start = row_ptr [row];
        int row_end = row_ptr [row +1];

        for (int j = row_start ; j < row_end ; j++)
            dotProduct += values [j] * x[col_ind[j]];

        y[ row ] += dotProduct;
    }
}


double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(int **row_ptr, int **col_ind, float **values, const char *filename, int *num_rows, int *num_cols, int *num_vals) {
    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    
    // Get number of rows, columns, and non-zero values
    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");
    
    int *row_ptr_t = (int *) malloc((*num_rows + 1) * sizeof(int));
    int *col_ind_t = (int *) malloc(*num_vals * sizeof(int));
    float *values_t = (float *) malloc(*num_vals * sizeof(float));
    
    // Collect occurances of each row for determining the indices of row_ptr
    int *row_occurances = (int *) malloc(*num_rows * sizeof(int));
    for (int i = 0; i < *num_rows; i++) {
        row_occurances[i] = 0;
    }
    
    int row, column;
    float value;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;
        
        row_occurances[row]++;
    }
    
    // Set row_ptr
    int index = 0;
    for (int i = 0; i < *num_rows; i++) {
        row_ptr_t[i] = index;
        index += row_occurances[i];
    }
    row_ptr_t[*num_rows] = *num_vals;
    free(row_occurances);
    
    // Set the file position to the beginning of the file
    rewind(file);
    
    // Read the file again, save column indices and values
    for (int i = 0; i < *num_vals; i++) {
        col_ind_t[i] = -1;
    }
    
    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");
    
    int i = 0;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
        row--;
        column--;
        
        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1) {
            i++;
        }
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        
        i = 0;
    }
    
    fclose(file);
    
    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;
}

// CPU implementation of SPMV using CSR, DO NOT CHANGE THIS
void spmv_csr_sw(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const float *x, float *y) {
    for (int i = 0; i < num_rows; i++) {
        float dotProduct = 0;
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            dotProduct += values[j] * x[col_ind[j]];
        }
        
        y[i] = dotProduct;
    }
}

int main(int argc, const char * argv[]) {

    if (argc != 3) {
        printf("Usage: ./exec matrix_file number_threads");
        return 0;
    }
    
    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values;
    
    const char *filename = argv[1];
    int th = atoi(argv[2]);

    double start, end;
    
    read_matrix(&row_ptr, &col_ind, &values, filename, &num_rows, &num_cols, &num_vals);
    
    float *x = (float *) malloc(num_rows * sizeof(float));
    float *y_sw = (float *) malloc(num_rows * sizeof(float));
    
    // Generate a random vector

    srand(time(NULL));
    for (int i = 0; i < num_rows; i++) {
        x[i] = (rand()%100)/(rand()%100+1); //the number we use to divide cannot be 0, that's the reason of the +1
    }
    
    // Compute in sw
    start = get_time();
    spmv_csr_sw(row_ptr, col_ind, values, num_rows, x, y_sw);
    end = get_time();

    // Print time
    printf("SPMV Time CPU: %.10lf\n", end - start);
    
    // Create variables for device operation
    int *d_row_ptr, *d_col_ind;
    float *d_values, *d_x, *d_y; 
    
    // Allocate memory device for the variables
    CHECK(cudaMalloc(&d_row_ptr, (num_rows+1)*sizeof(int)));
    CHECK(cudaMalloc(&d_col_ind, num_vals * sizeof(int)));
    CHECK(cudaMalloc(&d_values, num_vals * sizeof(float)));
    CHECK(cudaMalloc(&d_x, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&d_y, num_rows * sizeof(float)));
    
    // Copy data between host and device
    CHECK(cudaMemcpy(d_row_ptr, row_ptr, (num_rows+1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_col_ind, col_ind, num_vals*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_values, values, (num_vals)*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, x, (num_rows)*sizeof(float), cudaMemcpyHostToDevice));

    // Compute SPMV in Kernel 
    start = get_time();
    dim3 blocksPerGrid(num_rows/th, 1, 1);
    dim3 threadsPerBlock(th, 1, 1);
    spmv_csr<<<blocksPerGrid, threadsPerBlock>>> (num_rows, d_row_ptr, d_col_ind, d_values, d_x, d_y);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    end = get_time();

    printf("SPMV Time GPU: %.10lf\n", end - start);
    
    // Copy result from device to host
    CHECK(cudaMemcpy(y_sw, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free memory on the device
    CHECK(cudaFree(d_row_ptr));
    CHECK(cudaFree(d_col_ind));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));

    // Free memory on the host
    free(row_ptr);
    free(col_ind);
    free(values);
    free(y_sw);
    
    return 0;
}