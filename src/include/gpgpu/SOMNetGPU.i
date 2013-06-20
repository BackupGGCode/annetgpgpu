%{
#include "gpgpu/SOMNetGPU.h"
%}

%ignore ANNGPGPU::SOMNetGPU::SetDistFunction(const ANN::DistFunction *);

%include "gpgpu/SOMNetGPU.h" 