#!/usr/bin/python

import numpy
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

a = numpy.random.randn(4,4).astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)

func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print (a_doubled)
print 
print (a)

dev = pycuda.autoinit.device

print "{}: max block: {}, max grid: {}".format(dev.name(), dev.MAX_BLOCK_DIM_Y, dev.MAX_GRID_DIM_X)

print