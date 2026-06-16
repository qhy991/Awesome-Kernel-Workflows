ARCH ?= native
NVCC ?= nvcc

kernel_bench: main.cu
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

clean:
	rm -f kernel_bench
