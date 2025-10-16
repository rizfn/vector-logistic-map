#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <filesystem>

// Default parameters
constexpr int DEFAULT_N = 256;
constexpr float DEFAULT_A = 3.9f;
constexpr float DEFAULT_EPSILON = 0.1f;
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

// Add Gram-Schmidt orthonormalization on host
void gramSchmidt(std::vector<float>& v, int N, int n_vecs) {
    // v is a flat array of size N * n_vecs (column-major)
    for (int k = 0; k < n_vecs; ++k) {
        // Normalize k-th vector
        float norm = 0.0f;
        for (int i = 0; i < N; ++i) norm += v[k * N + i] * v[k * N + i];
        norm = std::sqrt(norm);
        for (int i = 0; i < N; ++i) v[k * N + i] /= norm;
        // Remove projections from later vectors
        for (int l = k + 1; l < n_vecs; ++l) {
            float dot = 0.0f;
            for (int i = 0; i < N; ++i) dot += v[k * N + i] * v[l * N + i];
            for (int i = 0; i < N; ++i) v[l * N + i] -= dot * v[k * N + i];
        }
    }
}

// Kernel to evolve tangent vectors: v_new[i][k] = sum_j J[i][j] * v_old[j][k]
__global__ void updateTangentVectors(const float* x, const float* v_old, float* v_new, const float* A, int N, int n_vecs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && k < n_vecs) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float J_ij = A[i * N + j] * (1.0f - 2.0f * x[j]);
            sum += J_ij * v_old[j + k * N];
        }
        v_new[i + k * N] = sum;
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments or use defaults
    int N = (argc > 1) ? std::atoi(argv[1]) : DEFAULT_N;
    float a = (argc > 2) ? std::atof(argv[2]) : DEFAULT_A;
    float epsilon = (argc > 3) ? std::atof(argv[3]) : DEFAULT_EPSILON;
    int N_steps = (argc > 4) ? std::atoi(argv[4]) : DEFAULT_N_STEPS;

    // Lyapunov spectrum parameters
    int n_transient = 4096; // Disregard these steps
    int n_measure = 2048;   // Steps to measure Lyapunov exponents
    int n_vecs = N;         // Full spectrum

    // Initialize lattice
    std::vector<float> x(N);
    for (int i = 0; i < N; i++) x[i] = 0.5f + 0.1f * (float)i / N;

    // Allocate tangent vectors (host and device)
    std::vector<float> v(N * n_vecs, 0.0f);
    for (int k = 0; k < n_vecs; ++k) v[k * N + k] = 1.0f; // Identity basis

    float *d_x_old, *d_x_new, *d_A, *d_v_old, *d_v_new;
    cudaMalloc(&d_x_old, N * sizeof(float));
    cudaMalloc(&d_x_new, N * sizeof(float));
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_v_old, N * n_vecs * sizeof(float));
    cudaMalloc(&d_v_new, N * n_vecs * sizeof(float));

    cudaMemcpy(d_x_old, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_old, v.data(), N * n_vecs * sizeof(float), cudaMemcpyHostToDevice);

    // Create coupling matrix on GPU
    dim3 matrixBlock(16, 16);
    dim3 matrixGrid((N + 15) / 16, (N + 15) / 16);
    createCouplingMatrixKernel<<<matrixGrid, matrixBlock>>>(d_A, N, a, epsilon);
    cudaDeviceSynchronize();
    
    // CUDA kernel configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    dim3 tangentBlock(16, 16);
    dim3 tangentGrid((N + 15) / 16, (n_vecs + 15) / 16);

    // Transient evolution (no Lyapunov measurement)
    for (int step = 0; step < n_transient; ++step) {
        updateLattice<<<gridSize, blockSize>>>(d_x_old, d_x_new, d_A, N);
        cudaDeviceSynchronize();
        std::swap(d_x_old, d_x_new);
    }

    // Lyapunov measurement
    std::vector<double> lyap_sum(n_vecs, 0.0);
    for (int step = 0; step < n_measure; ++step) {
        // Evolve lattice
        updateLattice<<<gridSize, blockSize>>>(d_x_old, d_x_new, d_A, N);
        cudaDeviceSynchronize();
        std::swap(d_x_old, d_x_new);

        // Evolve tangent vectors
        updateTangentVectors<<<tangentGrid, tangentBlock>>>(d_x_old, d_v_old, d_v_new, d_A, N, n_vecs);
        cudaDeviceSynchronize();

        // Copy tangent vectors to host for Gram-Schmidt and norm calculation
        cudaMemcpy(v.data(), d_v_new, N * n_vecs * sizeof(float), cudaMemcpyDeviceToHost);

        // Gram-Schmidt orthonormalization and accumulate log norms
        for (int k = 0; k < n_vecs; ++k) {
            double norm = 0.0;
            for (int i = 0; i < N; ++i) norm += v[k * N + i] * v[k * N + i];
            norm = std::sqrt(norm);
            lyap_sum[k] += std::log(norm);
            for (int i = 0; i < N; ++i) v[k * N + i] /= norm;
        }
        gramSchmidt(v, N, n_vecs);

        // Copy orthonormalized tangent vectors back to device
        cudaMemcpy(d_v_old, v.data(), N * n_vecs * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Compute Lyapunov exponents
    std::vector<double> lyap(n_vecs);
    for (int k = 0; k < n_vecs; ++k) lyap[k] = lyap_sum[k] / n_measure;

    // Output Lyapunov spectrum
    std::filesystem::path exe_path = std::filesystem::canonical("/proc/self/exe");
    std::filesystem::path exe_dir = exe_path.parent_path();
    std::filesystem::path output_dir = exe_dir / "outputs" / "lyapunov";
    std::filesystem::create_directories(output_dir);

    std::ostringstream fname;
    fname << "N_" << N << "_a_" << a << "_e_" << epsilon << ".tsv";
    std::filesystem::path output_file = output_dir / fname.str();

    std::ofstream outfile(output_file);
    outfile << std::setprecision(10);
    for (int k = 0; k < n_vecs; ++k) {
        outfile << lyap[k];
        if (k < n_vecs - 1) outfile << "\t";
    }
    outfile << "\n";
    outfile.close();

    // Free device memory
    cudaFree(d_x_old);
    cudaFree(d_x_new);
    cudaFree(d_A);
    cudaFree(d_v_old);
    cudaFree(d_v_new);

    std::cout << "Lyapunov spectrum written to " << output_file.string() << std::endl;
    return 0;
}
