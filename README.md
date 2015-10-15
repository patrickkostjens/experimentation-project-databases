# Experimentation Project Databases

Experimental implementation of database queries on GPUs. 

The idea is to use CUDA to process (parts of) queries using the GPU. There will also be CPU implementations of the same operations to make a comparison between the two.

The Visual Studio Solution works fully with Visual Studio 2013 and CUDA installed. However, without CUDA, the CPU parts can still be executed by starting the MainCPU project.

A file called 'lineitem.tbl' is required to be in the 'code' folder. This file can be generated using the DBGEN tool of the TPC-H benchmark.
