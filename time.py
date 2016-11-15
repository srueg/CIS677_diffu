#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
import pycuda.driver as cuda
import pycuda.autoinit
import math
from pycuda.compiler import SourceModule


INIT_TEMP = numpy.float32(23.0)
STEPS = 2000000
BLOCKS = 1024
ROD_LENGTH = 2000
TIME_PLOTS = 20

a = numpy.full((1, ROD_LENGTH), INIT_TEMP, dtype=numpy.float32)
result = numpy.copy(a)
b = numpy.empty_like(a)

a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

mod = SourceModule("""
  # define SOURCE_TEMP 100.00

  __global__ void calc(float *a, float *b, int length)
  {
      int idx =  blockIdx.x * blockDim.x + threadIdx.x;

      if(idx >= length){
        return;
      }else if(idx == 0){
        // left boundary
        b[idx] = (a[idx+1] + SOURCE_TEMP) / 2;
      }else if (idx == length-1){
        // right boundary
        b[idx] = (a[idx] + a[idx-1]) / 2;
      }else{
        b[idx] = (a[idx+1] + a[idx-1]) / 2;
      }
  }
  """)

# create two timers so we can speed-test each approach
start = cuda.Event()
end = cuda.Event()

func = mod.get_function("calc")

block = (BLOCKS, 1, 1)
grid = (int(math.ceil(float(ROD_LENGTH) / BLOCKS)), 1)
start.record()
for i in range(0, STEPS):
    func(a_gpu, b_gpu, block=block, grid=grid)
    cuda.memcpy_dtod(a_gpu, b_gpu, a.nbytes)

cuda.memcpy_dtoh(b, b_gpu)

end.record()
start.synchronize()
end.synchronize()
secs = start.time_till(end) * 1e-3

dev = pycuda.autoinit.device

print "It took {}s on {}".format(secs, dev.name())

print b
