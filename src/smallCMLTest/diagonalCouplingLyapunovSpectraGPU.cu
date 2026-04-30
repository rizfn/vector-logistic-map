#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <random>

// Default parameters
constexpr int DEFAULT_N = 1024;
constexpr float DEFAULT_A = 3.9f;
constexpr float DEFAULT_EPSILON = 0.1f;
constexpr int DEFAULT_N_TRANSIENT = 16384;
constexpr int DEFAULT_N_MEASURE = 2048;
constexpr int DEFAULT_SIM_NO = 0;

// CUDA kernel for creating the coupling matrix on GPU
__global__ void createCouplingMatrixKernel(float* A, int N, float a, float epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < N && j < N) {
        if (i == j) {
            A[i * N + j] = a * (1.0f - 2.0f * epsilon);
        } else if ((j == (i + 1) % N) || (j == (i - 1 + N) % N)) {
            A[i * N + j] = a * epsilon;
        } else {
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

// Kernel to evolve tangent vectors
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

// Kernel to extract diagonal norms from R matrix (upper triangular from QR)
__global__ void extractDiagonalNorms(const float* R, double* log_norms, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float r_ii = R[i * N + i];
        log_norms[i] += log(fabs(r_ii));
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments
    int N = (argc > 1) ? std::atoi(argv[1]) : DEFAULT_N;
    float a = (argc > 2) ? std::atof(argv[2]) : DEFAULT_A;
    float epsilon = (argc > 3) ? std::atof(argv[3]) : DEFAULT_EPSILON;
    int simNo = (argc > 4) ? std::atoi(argv[4]) : DEFAULT_SIM_NO;
    int n_transient = DEFAULT_N_TRANSIENT;
    int n_measure = DEFAULT_N_MEASURE;

    // Initialize lattice with random initial conditions
    std::mt19937 rng(simNo + 42);
    std::uniform_real_distribution<float> dist(0.25f, 0.75f);
    std::vector<float> x(N);
    for (int i = 0; i < N; i++) x[i] = dist(rng);

    // Initialize tangent vectors (identity basis)
    std::vector<float> v(N * N, 0.0f);
    for (int k = 0; k < N; ++k) v[k * N + k] = 1.0f;

    // Allocate device memory
    float *d_x_old, *d_x_new, *d_A, *d_v_old, *d_v_new;
    double *d_log_norms;
    cudaMalloc(&d_x_old, N * sizeof(float));
    cudaMalloc(&d_x_new, N * sizeof(float));
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_v_old, N * N * sizeof(float));
    cudaMalloc(&d_v_new, N * N * sizeof(float));
    cudaMalloc(&d_log_norms, N * sizeof(double));
    cudaMemset(d_log_norms, 0, N * sizeof(double));

    cudaMemcpy(d_x_old, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_old, v.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Create coupling matrix
    dim3 matrixBlock(16, 16);
    dim3 matrixGrid((N + 15) / 16, (N + 15) / 16);
    createCouplingMatrixKernel<<<matrixGrid, matrixBlock>>>(d_A, N, a, epsilon);
    cudaDeviceSynchronize();

    // CUDA kernel configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    dim3 tangentBlock(16, 16);
    dim3 tangentGrid((N + 15) / 16, (N + 15) / 16);

    // cuSOLVER setup for QR decomposition
    cusolverDnHandle_t cusolverH = nullptr;
    cusolverDnCreate(&cusolverH);
    
    int *d_info;
    float *d_tau, *d_work;
    int lwork = 0;
    cudaMalloc(&d_info, sizeof(int));
    cudaMalloc(&d_tau, N * sizeof(float));
    
    // Query workspace size
    cusolverDnSgeqrf_bufferSize(cusolverH, N, N, d_v_old, N, &lwork);
    cudaMalloc(&d_work, lwork * sizeof(float));

    // Transient evolution
    for (int step = 0; step < n_transient; ++step) {
        updateLattice<<<gridSize, blockSize>>>(d_x_old, d_x_new, d_A, N);
        cudaDeviceSynchronize();
        std::swap(d_x_old, d_x_new);
    }

    // Lyapunov measurement with QR orthonormalization
    for (int step = 0; step < n_measure; ++step) {
        // Evolve reference
        updateLattice<<<gridSize, blockSize>>>(d_x_old, d_x_new, d_A, N);
        
        // Evolve tangent vectors
        updateTangentVectors<<<tangentGrid, tangentBlock>>>(d_x_old, d_v_old, d_v_new, d_A, N, N);
        cudaDeviceSynchronize();
        
        // QR decomposition: v_new = Q * R (in-place, v_new becomes R on upper triangle)
        cusolverDnSgeqrf(cusolverH, N, N, d_v_new, N, d_tau, d_work, lwork, d_info);
        cudaDeviceSynchronize();
        
        // Accumulate log of diagonal norms (R diagonal)
        extractDiagonalNorms<<<gridSize, blockSize>>>(d_v_new, d_log_norms, N);
        
        // Reconstruct Q from Householder reflectors for next iteration
        cusolverDnSorgqr(cusolverH, N, N, N, d_v_new, N, d_tau, d_work, lwork, d_info);
        cudaDeviceSynchronize();
        
        std::swap(d_x_old, d_x_new);
        std::swap(d_v_old, d_v_new);
    }

    // Copy Lyapunov sums to host
    std::vector<double> log_norms(N);
    cudaMemcpy(log_norms.data(), d_log_norms, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Copy coupling matrix A to host
    std::vector<float> A_host(N * N);
    cudaMemcpy(A_host.data(), d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute Lyapunov exponents
    std::vector<double> lyap(N);
    for (int k = 0; k < N; ++k) lyap[k] = log_norms[k] / n_measure;

    // Output Lyapunov spectrum
    std::filesystem::path exe_path = std::filesystem::canonical("/proc/self/exe");
    std::filesystem::path exe_dir = exe_path.parent_path();
    std::filesystem::path output_dir = exe_dir / "outputs" / "lyapunov_gpu";
    std::filesystem::create_directories(output_dir);

    std::ostringstream fname;
    fname << "N_" << N << "_a_" << a << "_e_" << epsilon << "_sim_" << simNo << ".tsv";
    std::filesystem::path output_file = output_dir / fname.str();

    std::ofstream outfile(output_file);
    outfile << std::setprecision(10);
    for (int k = 0; k < N; ++k) {
        outfile << lyap[k];
        if (k < N - 1) outfile << "\t";
    }
    outfile << "\n";
    outfile.close();

    // Save coupling matrix A (binary, once per parameter set)
    std::filesystem::path A_dir = exe_dir / "outputs" / "coupling_matrices";
    std::filesystem::create_directories(A_dir);
    
    std::ostringstream A_fname;
    A_fname << "A_N_" << N << "_a_" << a << "_e_" << epsilon << ".bin";
    std::filesystem::path A_file = A_dir / A_fname.str();
    
    std::ofstream A_out(A_file, std::ios::binary);
    A_out.write(reinterpret_cast<const char*>(A_host.data()), N * N * sizeof(float));
    A_out.close();

    // Cleanup
    cudaFree(d_x_old);
    cudaFree(d_x_new);
    cudaFree(d_A);
    cudaFree(d_v_old);
    cudaFree(d_v_new);
    cudaFree(d_log_norms);
    cudaFree(d_tau);
    cudaFree(d_work);
    cudaFree(d_info);
    cusolverDnDestroy(cusolverH);

    std::cout << "Lyapunov spectrum written to " << output_file.string() << std::endl;
    std::cout << "Coupling matrix A written to " << A_file.string() << std::endl;
    return 0;
}
