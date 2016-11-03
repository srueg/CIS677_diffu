#!/usr/bin/python

import numpy
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

INIT_TEMP = numpy.float32(23.0)

a = numpy.full((10,10), INIT_TEMP, dtype=numpy.float32)

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    a[idx] *= 2;
  }
  """)

# create two timers so we can speed-test each approach
start = cuda.Event()
end = cuda.Event()

func = mod.get_function("doublify")

grid = (1, 1)
block = a.shape + (1,)
func.prepare("P")

start.record()
func.prepared_call(grid, block, a_gpu)
end.record()
secs = start.time_till(end)*1e-3

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print (a_doubled)
print 
print (a)

dev = pycuda.autoinit.device

print "It took {}s on {}".format(secs, dev.name())

print