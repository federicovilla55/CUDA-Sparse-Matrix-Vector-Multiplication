## Introduction
<p align="justify">
This repository contains three different implementations of sparse matrix-vector multiplication. One implementation runs in C and runs on a single thread, the other two are in CUDA and runs on multiple GPU threads.
A more detailed description of the work can be found in the <em>Sparse_Matrix_Vector_Multiplication_Report.pdf</em> file.
</p>

## Usage

### Compilation

To compile the three files you just have to run:
```
make
```
Within the scope of this folder.

To compile the three files, C and CUDA compilers are needed.

To run each file after they have been compiled, just navigate in the <em>spmv</em> directory and run them with the command: `./FileToExecute`.

### Repository Tree
```
├── Makefile
├── README.md
├── Sparse_Matrix_Vector_Multiplication_Report.pdf
└── spmv
    ├── simple-spmv.cu
    ├── spmv-csr.c
    └── spmv-csr.cu
```
The three implementations of the sparse matrix-vector multiplication are:
-  <em>spmv-csr.c</em>, a C single thread CPU implementation;
-  <em>simple-spmv.cu</em>, a CUDA one thread per matrix row implementation;
-  <em>spmv-csr.c</em>, a CUDA one 32-thread warp per matrix row.