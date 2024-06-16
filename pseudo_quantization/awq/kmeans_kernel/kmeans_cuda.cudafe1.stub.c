#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "kmeans_cuda.fatbin.c"
extern void __device_stub__Z13kmeans_kernelPfS_PiS_S_iiS_S0_(float *, float *, int *, float *, float *, int, int, float *, int *);
static void __device_stub__ZN3cub11EmptyKernelIvEEvv(void);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z13kmeans_kernelPfS_PiS_S_iiS_S0_(float *__par0, float *__par1, int *__par2, float *__par3, float *__par4, int __par5, int __par6, float *__par7, int *__par8){__cudaLaunchPrologue(9);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 32UL);__cudaSetupArgSimple(__par5, 40UL);__cudaSetupArgSimple(__par6, 44UL);__cudaSetupArgSimple(__par7, 48UL);__cudaSetupArgSimple(__par8, 56UL);__cudaLaunch(((char *)((void ( *)(float *, float *, int *, float *, float *, int, int, float *, int *))kmeans_kernel)));}
# 15 "kmeans_cuda.cu"
void kmeans_kernel( float *__cuda_0,float *__cuda_1,int *__cuda_2,float *__cuda_3,float *__cuda_4,int __cuda_5,int __cuda_6,float *__cuda_7,int *__cuda_8)
# 15 "kmeans_cuda.cu"
{__device_stub__Z13kmeans_kernelPfS_PiS_S_iiS_S0_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8);
# 92 "kmeans_cuda.cu"
}
# 1 "kmeans_cuda.cudafe1.stub.c"
static void __device_stub__ZN3cub11EmptyKernelIvEEvv(void) {  __cudaLaunchPrologue(1); __cudaLaunch(((char *)((void ( *)(void))cub::EmptyKernel<void> ))); }namespace cub{

template<> __specialization_static void __wrapper__device_stub_EmptyKernel<void>(void){__device_stub__ZN3cub11EmptyKernelIvEEvv();}}
static void __nv_cudaEntityRegisterCallback( void **__T60) {  __nv_dummy_param_ref(__T60); __nv_save_fatbinhandle_for_managed_rt(__T60); __cudaRegisterEntry(__T60, ((void ( *)(void))cub::EmptyKernel<void> ), _ZN3cub11EmptyKernelIvEEvv, (-1)); __cudaRegisterEntry(__T60, ((void ( *)(float *, float *, int *, float *, float *, int, int, float *, int *))kmeans_kernel), _Z13kmeans_kernelPfS_PiS_S_iiS_S0_, (-1)); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust6system6detail10sequential3seqE,::thrust::system::detail::sequential::seq), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust6system3cpp3parE,::thrust::system::cpp::par), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust8cuda_cub3parE,::thrust::cuda_cub::par), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust12placeholders2_1E,::thrust::placeholders::_1), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust12placeholders2_2E,::thrust::placeholders::_2), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust12placeholders2_3E,::thrust::placeholders::_3), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust12placeholders2_4E,::thrust::placeholders::_4), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust12placeholders2_5E,::thrust::placeholders::_5), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust12placeholders2_6E,::thrust::placeholders::_6), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust12placeholders2_7E,::thrust::placeholders::_7), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust12placeholders2_8E,::thrust::placeholders::_8), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust12placeholders2_9E,::thrust::placeholders::_9), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust12placeholders3_10E,::thrust::placeholders::_10), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust3seqE,::thrust::seq), 0, 1UL, 0, 0); __cudaRegisterVariable(__T60, __shadow_var(_ZN45_INTERNAL_40c821d1_14_kmeans_cuda_cu_620c01f56thrust6deviceE,::thrust::device), 0, 1UL, 0, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
