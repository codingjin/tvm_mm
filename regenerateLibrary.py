import os
import time
import numpy as np
import tvm
from tvm import te, meta_schedule
import sys
import shutil
import argparse

def _matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
    )
    te_func = te.create_prim_func([A, B, matmul]).with_attr({"global_symbol": "main"})
    return tvm.IRModule({"main": te_func})

def generateFile(N, K, M, cpu_model, threadnum):
    mod = _matmul(N, K, M, "float32")
    if cpu_model == "r9":
        attr = "+avx512f"
        mcpu = "znver4"
    elif cpu_model == "rt":
        attr = "+avx2"
        mcpu = "znver2"
    else:
        attr = "+avx2"
        mcpu = "native"

    work_dir = f"./{cpu_model}/{threadnum}"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)
    target = f"llvm -num-cores {threadnum} -mcpu={mcpu} -mattr={attr}"
    database = meta_schedule.tune_tir(
        mod=mod,
        target=target,
        #max_trials_global=4,
        #num_trials_per_iter=1,
        max_trials_global=1000,
        num_trials_per_iter=64,
        work_dir=work_dir,
    )

    sch = meta_schedule.tir_integration.compile_tir(database, mod, target) 
    print("Best Schedule", flush=True)
    print(sch.trace.as_python())
    print(str(sch.mod))

    lib = tvm.build(sch.mod, target)
    with tvm.transform.PassContext(config={"tir.disable_assert":False}):
        with open(f"{work_dir}/{N}_{K}_{M}_source.txt", 'w') as f:
            print(lib.get_source(), flush=True, file=f)
            # print("Assembly Code: ", lib.get_source("asm"), flush=True, file=f)
            # print("C Code: ", lib.get_source("c"), flush=True, file=f)
    lib.export_library(f"{work_dir}/{N}_{K}_{M}.so")


def main():
    parser = argparse.ArgumentParser(
        description=".so file program generater, with user input arguments, e.g. python regenerateLibrary.py --cpu i7 --threadnum=8 --N=1024 --K=1024 --M=1024"
    )
    parser.add_argument(
        "--cpu",
        type=str,
        required=True,
        help="CPU model (e.g. i7, r9, rt)"
    )
    parser.add_argument(
        "--threadnum",
        type=int,
        required=True,
        help="Number of threads to set"
    )
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="First dimension size"
    )
    parser.add_argument(
        "--K",
        type=int,
        required=True,
        help="Middle dimenstion size"
    )
    parser.add_argument(
        "--M",
        type=int,
        required=True,
        help="Last dimension size"
    )
    args = parser.parse_args()
    print(f"CPU model: {args.cpu}")
    print(f"ThreadNum: {args.threadnum}")
    print(f"N * K * M: {args.N} * {args.K} * {args.M}")

    generateFile(args.N, args.K, args.M, args.cpu, args.threadnum)


if __name__ == '__main__':
    main()

