#!/bin/sh
glslangValidator -V kernel1D_MakeEyeRay.comp -o kernel1D_MakeEyeRay.comp.spv -DGLSL -I.. -I/home/frol/PROG/HydraRepos/HydraCore3/external/LiteScene -I/home/frol/PROG/HydraRepos/HydraCore3/external/LiteMath -I/home/frol/PROG/kernel_slicer/apps/LiteMathAux -I/home/frol/PROG/kernel_slicer/apps/LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V kernel1D_ContribSample.comp -o kernel1D_ContribSample.comp.spv -DGLSL -I.. -I/home/frol/PROG/HydraRepos/HydraCore3/external/LiteScene -I/home/frol/PROG/HydraRepos/HydraCore3/external/LiteMath -I/home/frol/PROG/kernel_slicer/apps/LiteMathAux -I/home/frol/PROG/kernel_slicer/apps/LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
