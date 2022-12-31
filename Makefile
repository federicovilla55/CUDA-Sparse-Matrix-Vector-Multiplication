CC 			=	gcc
NV 			= 	nvcc
CFLAGS		=	-O3 -Wunused-result
NVCCFLAGS	=	-O3
PROG		=	spmv/spmv-cpu spmv/spmv-gpu spmv/spmv-naive

all:$(PROG)

spmv/spmv-cpu: spmv/spmv-csr.c
	$(CC) $(CFLAGS) $^ -o $@ 

spmv/spmv-gpu: spmv/spmv-csr.cu
	$(NV) $(NVCCFLAGS) $^ -o $@

spmv/spmv-naive: spmv/simple-spmv.cu
	$(NV) $(NVCCFLAGS) $^ -o $@


.PHONY:clean
clean:
	rm -f $(PROG)