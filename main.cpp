#include <stdlib.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <string>
#include <random>
#include <cmath>
#include <fstream>
#include <cstring>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

struct arguments {
    int N;
    int K;
    int M;
    int ThreadNum;
    std::string CPUModel;
};

const int WARMUP = 10;
const int RUNS = 100;
const double ERR = 0.1;

void printResults(const std::string& name, const std::vector<double>& results, const double FLOPs)
{
    double total = std::accumulate(results.begin(), results.end(), 0.0);
    double avg = total/results.size();
    double median = results[results.size()/2];
    double min = results[0];
    double dev = 0.0;

    for (const auto re : results)
        dev += (re - avg) * (re - avg);
    dev /= results.size();

    std::cout << "=== " << name << " ===" << std::endl;
    std::cout << "Took " << total << " seconds for " << RUNS << " runs. " << WARMUP << " warmups\n";
    std::cout << "Med " << median << "\tMed (" << FLOPs/1.0e9/median << " GFLOPS)\n";
    std::cout << "Min " << min << "\tMax (" << FLOPs/1.0e9/min << " GFLOPS)\n";
    std::cout << "Avg " << avg << "\tAvg (" << FLOPs/1.0e9/avg << " GFLOPS)\n";
    std::cout << dev << " Dev\n" << std::endl;
}

tvm::ffi::Function getTVMFunc(const struct arguments& args) {
  // Check to see if the library is there
  //std::string fileName = std::format("./{}/{}/{}_{}_{}.so", args.CPUModel, args.ThreadNum, args.N, args.K, args.M);
  std::string fileName = "./" + args.CPUModel + "/" + std::to_string(args.ThreadNum) + "/" 
                          + std::to_string(args.N) + "_" + std::to_string(args.K) + "_" 
                          + std::to_string(args.M) + ".so";

  if (std::filesystem::exists(fileName)) {
    tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile(fileName);
    tvm::ffi::Function func = mod.GetFunction("main");
    if (func.operator->() != nullptr) {
      return func;
    }
    std::cout << "Couldn't find main inside " << fileName << "." << std::endl;
  }

  // Regenerate library
  //std::string regenerate_command = std::format("python3 ./regenerateLibrary.py --cpu {} --threadnum {} --N {} --K {} --M {}", 
  //                                  args.CPUModel, args.ThreadNum, args.N, args.K, args.M);
  std::string regenerate_command = "python3 ./regenerateLibrary.py --cpu " + args.CPUModel
                                    + " --threadnum " + std::to_string(args.ThreadNum)
                                    + " --N " + std::to_string(args.N)
                                    + " --K " + std::to_string(args.K)
                                    + " --M " + std::to_string(args.M);
  std::cout << "Regenerating library" << std::endl;
  std::cout << regenerate_command << std::endl;
  system(regenerate_command.c_str());

  if (std::filesystem::exists(fileName)) {
    tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile(fileName);
    tvm::ffi::Function func = mod.GetFunction("main");
    if (func.operator->() != nullptr) {
      return func;
    }
    std::cout << "Couldn't find main inside " << fileName << "." << std::endl;
  } else {
    std::cout << "Couldn't find " << fileName << "." << std::endl;
  }
  exit(1);
}

int main(int argc, char* argv[]) {

    if (argc != 6) {
        std::cerr << "Invalid Usage!\n";
        std::cout << "Usage: ./tvm_sgemm cpu_model N K M ThreadNum" << std::endl;
        exit(1);
    }
    struct arguments args;
    args.CPUModel = argv[1];
    args.N = std::stoi(argv[2]);
    args.K = std::stoi(argv[3]);
    args.M = std::stoi(argv[4]);
    args.ThreadNum = std::stoi(argv[5]);
    //std::cout << "cpu_model=" << args.CPUModel << ", N=" << args.N << ", K=" << args.K << ", M=" << args.M << ", ThreadNum=" << args.ThreadNum << std::endl;

    tvm::ffi::Function func = getTVMFunc(args);
    int N = args.N;
    int K = args.K;
    int M = args.M;
    tvm::runtime::NDArray A = tvm::runtime::NDArray::Empty({N, K}, DLDataType{kDLFloat, 32, 1}, DLDevice{kDLCPU, 0});
    tvm::runtime::NDArray B = tvm::runtime::NDArray::Empty({K, M}, DLDataType{kDLFloat, 32, 1}, DLDevice{kDLCPU, 0});
    tvm::runtime::NDArray C = tvm::runtime::NDArray::Empty({N, M}, DLDataType{kDLFloat, 32, 1}, DLDevice{kDLCPU, 0});
    float* A_data = static_cast<float*>(A->data);
    float* B_data = static_cast<float*>(B->data);
    float* C_data = static_cast<float*>(C->data);
    std::mt19937 engine{137};
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N*K; i++) A_data[i] = dist(engine);
    for (int i = 0; i < K*M; i++) B_data[i] = dist(engine);
    std::memset(C_data, 0, sizeof(float) * std::size_t(N) * std::size_t(M));

    /*
    func(A, B, C);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            float tmp = 0.0;
            for (int k = 0; k < K; ++k)  tmp += A_data[i*K+k] * B_data[k*M+j];
            if (std::abs(tmp - C_data[i*M+j]) > ERR) {
                std::cerr << "Correctness check failed!\n" << "N=" << N << " M=" << M << std::endl;
                std::cerr << "i=" << i << ", j=" << j << " A=" << A_data[i*M+j] << ", B=" << B_data[i*M+j] << std::endl;
                exit(1);
            }
        }
    }
    std::memset(C_data, 0, sizeof(float) * std::size_t(N) * std::size_t(M));
    //std::cout << "Correctness checking passed!" << std::endl;
    */
    
    for (int i = 0; i < WARMUP; ++i) {
        func(A, B, C);
        std::memset(C_data, 0, sizeof(float) * std::size_t(N) * std::size_t(M));
    }

    std::vector<double> results;
    for (int i = 0; i < RUNS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        results.push_back(duration.count());
        std::memset(C_data, 0, sizeof(float) * std::size_t(N) * std::size_t(M));
    }
    std::sort(results.begin(), results.end());
    std::string name = std::string("TVM CPUModel=") + args.CPUModel + " " + 
                        std::to_string(N) + "_" +
                        std::to_string(K) + "_" +
                        std::to_string(M) + " ThreadNum=" +
                        std::to_string(args.ThreadNum);
    double FLOPs = 2.0 * M * N * K;
    printResults(name, results, FLOPs);
    return 0;
}