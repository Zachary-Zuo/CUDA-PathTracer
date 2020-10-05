#pragma once

#include <stdio.h>
#include "ShaderStructs.h"
#include "helper_cuda.h"

void RunSineWaveKernel(cudaExternalSemaphore_t &extSemaphore, uint64_t &key, unsigned int timeoutMs,
                        size_t mesh_width, size_t mesh_height, DXVertex *cudaDevVertptr, cudaStream_t streamToRun);
DXVertex* cudaImportVertexBuffer(void*sharedHandle, cudaExternalMemory_t &externalMemory, int meshWidth, int meshHeight);
void cudaImportKeyedMutex(void*sharedHandle, cudaExternalSemaphore_t &extSemaphore);