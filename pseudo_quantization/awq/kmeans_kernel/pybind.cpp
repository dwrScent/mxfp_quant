#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "kmeans_cuda.h"
#include "weighted_kmeans_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kmeans_cuda_forward", &kmeans_cuda_forward, "our kmeans kernel");
    m.def("weighted_kmeans_cuda", &weighted_kmeans_cuda, "our weighted kmeans kernel");
}
