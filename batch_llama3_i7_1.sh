#!/bin/bash

make 

modelname="llama3"
cpumodel="i7"
threadnum="20"


mkdir -p ${modelname}/${cpumodel}/${threadnum}

./tvm_sgemm ${cpumodel} 4096 4096 128 ${threadnum} | tee ${modelname}/${cpumodel}/${threadnum}/4096_4096_128

./tvm_sgemm ${cpumodel} 128 4096 8192 ${threadnum} | tee ${modelname}/${cpumodel}/${threadnum}/128_4096_8192

./tvm_sgemm ${cpumodel} 128 8192 4096 ${threadnum} | tee ${modelname}/${cpumodel}/${threadnum}/128_8192_4096

./tvm_sgemm ${cpumodel} 4096 4096 4096 ${threadnum} | tee ${modelname}/${cpumodel}/${threadnum}/4096_4096_4096


./tvm_sgemm ${cpumodel} 4097 4097 129 ${threadnum} | tee ${modelname}/${cpumodel}/${threadnum}/4097_4097_129

./tvm_sgemm ${cpumodel} 129 4097 8193 ${threadnum} | tee ${modelname}/${cpumodel}/${threadnum}/129_4097_8193

./tvm_sgemm ${cpumodel} 129 8193 4097 ${threadnum} | tee ${modelname}/${cpumodel}/${threadnum}/129_8193_4097

./tvm_sgemm ${cpumodel} 4097 4097 4097 ${threadnum} | tee ${modelname}/${cpumodel}/${threadnum}/4097_4097_4097


./tvm_sgemm ${cpumodel} 4095 4095 127 ${threadnum} | tee ${modelname}/${cpumodel}/${threadnum}/4095_4095_127

./tvm_sgemm ${cpumodel} 127 4095 8191 ${threadnum} | tee ${modelname}/${cpumodel}/${threadnum}/127_4095_8191

./tvm_sgemm ${cpumodel} 127 8191 4095 ${threadnum} | tee ${modelname}/${cpumodel}/${threadnum}/127_8191_4095

./tvm_sgemm ${cpumodel} 4095 4095 4095 ${threadnum} | tee ${modelname}/${cpumodel}/${threadnum}/4095_4095_4095
