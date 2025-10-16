#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <filesystem>

// Default parameters
constexpr int DEFAULT_N = 256;
constexpr double DEFAULT_A = 3.9;
constexpr double DEFAULT_EPSILON = 0.01;
constexpr int DEFAULT_N_STEPS = 16384;

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : DEFAULT_N;
    double a = (argc > 2) ? std::atof(argv[2]) : DEFAULT_A;
    double epsilon = (argc > 3) ? std::atof(argv[3]) : DEFAULT_EPSILON;
    int N_steps = (argc > 4) ? std::atoi(argv[4]) : DEFAULT_N_STEPS;

    int n_recorded = (N_steps + 1) / 2;

    // Coupling matrix (row-major)
    std::vector<double> A(N * N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                A[i * N + j] = a * (1.0 - 2.0 * epsilon);
            } else if ((j == (i + 1) % N) || (j == (i - 1 + N) % N)) {
                A[i * N + j] = a * epsilon;
            }
        }
    }

    // Initial lattice
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = 0.5 + 0.1 * double(i) / N;

    std::vector<double> x_new(N);
    std::vector<double> output_data(n_recorded * N);

    int record_index = 0;
    for (int step = 0; step < N_steps; ++step) {
        if (step % 2 == 0) {
            for (int i = 0; i < N; ++i) {
                output_data[record_index * N + i] = x[i];
            }
            record_index++;
        }
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j) {
                sum += A[i * N + j] * x[j] * (1.0 - x[j]);
            }
            x_new[i] = sum;
        }
        x.swap(x_new);
    }

    std::filesystem::path output_dir = "outputs/timeseries";
    std::filesystem::create_directories(output_dir);

    std::ostringstream fname;
    fname << "N_" << N << "_a_" << a << "_e_" << epsilon << ".tsv";
    std::filesystem::path output_file = output_dir / fname.str();

    std::ofstream outfile(output_file);
    outfile << std::setprecision(10);
    for (int t = 0; t < n_recorded; ++t) {
        for (int i = 0; i < N; ++i) {
            outfile << output_data[t * N + i];
            if (i < N - 1) outfile << "\t";
        }
        outfile << "\n";
    }
    outfile.close();

    std::cout << "Simulation complete. Results written to " << output_file.string() << std::endl;
    return 0;
}
