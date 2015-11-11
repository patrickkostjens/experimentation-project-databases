# Experimentation Project Databases

Experimental implementation of database queries on GPUs. 

The idea is to use CUDA to process (parts of) queries using the GPU. There will also be CPU implementations of the same operations to make a comparison between the two.

The Visual Studio Solution works fully with Visual Studio 2013 and CUDA installed. However, without CUDA, the CPU parts can still be executed by starting the MainCPU project. To use CUDA's Unified Memory, the projects need to be compiled for the x64 platform.

Files called 'lineitem.tbl' and 'orders.tbl' are required to be in the 'code' folder. These files can be generated using the DBGEN tool of the TPC-H benchmark.
See: [TPC-H specification](http://www.tpc.org/tpch/spec/tpch2.7.0.pdf)
