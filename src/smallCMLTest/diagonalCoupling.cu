#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <filesystem>

// Default parameters
constexpr int DEFAULT_N = 256;
constexpr float DEFAULT_A = 3.9f;
constexpr float DEFAULT_EPSILON = 0.01f;
constexpr int DEFAULT_N_STEPS = 16384;

// CUDA kernel for creating the coupling matrix on GPU
__global__ void createCouplingMatrixKernel(float* A, int N, float a, float epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < N && j < N) {
        if (i == j) {
            // Diagonal: a(1 - 2*epsilon)
            A[i * N + j] = a * (1.0f - 2.0f * epsilon);
        } else if ((j == (i + 1) % N) || (j == (i - 1 + N) % N)) {
            // Periodic boundary: neighbors wrap around
            A[i * N + j] = a * epsilon;
        } else {
            // All other entries: 0
            A[i * N + j] = 0.0f;
        }
    }
}

// CUDA kernel for updating the lattice
__global__ void updateLattice(const float* x_old, float* x_new, const float* A, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float x_j = x_old[j];
            sum += A[i * N + j] * x_j * (1.0f - x_j);
        }
        x_new[i] = sum;
    }
}

// CUDA kernel for recording data
__global__ void recordData(const float* x, float* output, int N, int step_index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        output[step_index * N + i] = x[i];
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments or use defaults
    int N = (argc > 1) ? std::atoi(argv[1]) : DEFAULT_N;
    float a = (argc > 2) ? std::atof(argv[2]) : DEFAULT_A;
    float epsilon = (argc > 3) ? std::atof(argv[3]) : DEFAULT_EPSILON;
    int N_steps = (argc > 4) ? std::atoi(argv[4]) : DEFAULT_N_STEPS;
    
    // Calculate number of recorded steps
    int n_recorded = (N_steps + 1) / 2;  // Records at steps 0, 2, 4, ...
    
    // Initialize lattice with simple values
    std::vector<float> x(N);
    for (int i = 0; i < N; i++) {
        x[i] = 0.5f + 0.1f * (float)i / N;
    }
    
    // Allocate device memory
    float *d_x_old, *d_x_new, *d_A, *d_output;
    cudaMalloc(&d_x_old, N * sizeof(float));
    cudaMalloc(&d_x_new, N * sizeof(float));
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_output, n_recorded * N * sizeof(float));
    
    // Copy initial data to device
    cudaMemcpy(d_x_old, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create coupling matrix on GPU
    dim3 matrixBlock(16, 16);
    dim3 matrixGrid((N + 15) / 16, (N + 15) / 16);
    createCouplingMatrixKernel<<<matrixGrid, matrixBlock>>>(d_A, N, a, epsilon);
    cudaDeviceSynchronize();
    
    // CUDA kernel configuration for updates
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Evolution loop
    int record_index = 0;
    for (int step = 0; step < N_steps; step++) {
        // Record data every 2 steps
        if (step % 2 == 0) {
            recordData<<<gridSize, blockSize>>>(d_x_old, d_output, N, record_index);
            record_index++;
        }
        
        // Update lattice
        updateLattice<<<gridSize, blockSize>>>(d_x_old, d_x_new, d_A, N);
        cudaDeviceSynchronize();
        
        // Swap pointers for next iteration
        std::swap(d_x_old, d_x_new);
    }
    
    // Copy all recorded data from device to host at once
    std::vector<float> output_data(n_recorded * N);
    cudaMemcpy(output_data.data(), d_output, n_recorded * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Get executable directory
    std::filesystem::path exe_path = std::filesystem::canonical("/proc/self/exe");
    std::filesystem::path exe_dir = exe_path.parent_path();
    std::filesystem::path output_dir = exe_dir / "outputs" / "timeseries";
    
    // Create outputs/timeseries directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    
    // Format output filename with parameters
    std::ostringstream fname;
    fname << "N_" << N << "_a_" << a << "_e_" << epsilon << ".tsv";
    std::filesystem::path output_file = output_dir / fname.str();

    // Write to file
    std::ofstream outfile(output_file);
    for (int t = 0; t < n_recorded; t++) {
        for (int i = 0; i < N; i++) {
            outfile << output_data[t * N + i];
            if (i < N - 1) outfile << "\t";
        }
        outfile << "\n";
    }
    outfile.close();
    
    // Free device memory
    cudaFree(d_x_old);
    cudaFree(d_x_new);
    cudaFree(d_A);
    cudaFree(d_output);
    
    std::cout << "Simulation complete. Results written to " << output_file.string() << std::endl;
    
    return 0;
}
