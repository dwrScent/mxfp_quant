#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "wegihted_kmeans_cuda.fatbin.c"
extern void __device_stub__Z13kmeans_kernelPfS_S_PiS_S_iiiS_S_(float *, float *, float *, int *, float *, float *, int, int, int, float *, float *);
static void __device_stub__ZN3cub11EmptyKernelIvEEvv(void);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z13kmeans_kernelPfS_S_PiS_S_iiiS_S_(float *__par0, float *__par1, float *__par2, int *__par3, float *__par4, float *__par5, int __par6, int __par7, int __par8, float *__par9, float *__par10){__cudaLaunchPrologue(11);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 32UL);__cudaSetupArgSimple(__par5, 40UL);__cudaSetupArgSimple(__par6, 48UL);__cudaSetupArgSimple(__par7, 52UL);__cudaSetupArgSimple(__par8, 56UL);__cudaSetupArgSimple(__par9, 64UL);__cudaSetupArgSimple(__par10, 72UL);__cudaLaunch(((char *)((void ( *)(float *, float *, float *, int *, float *, float *, int, int, int, float *, float *))kmeans_kernel)));}
# 16 "wegihted_kmeans_cuda.cu"
void kmeans_kernel( float *__cuda_0,float *__cuda_1,float *__cuda_2,int *__cuda_3,float *__cuda_4,float *__cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,float *__cuda_9,float *__cuda_10)
# 16 "wegihted_kmeans_cuda.cu"
{__device_stub__Z13kmeans_kernelPfS_S_PiS_S_iiiS_S_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10);
# 93 "wegihted_kmeans_cuda.cu"
}
# 1 "wegihted_kmeans_cuda.cudafe1.stub.c"
static void __device_stub__ZN3cub11EmptyKernelIvEEvv(void) {  __cudaLaunchPrologue(1); __cudaLaunch(((char *)((void ( *)(void))cub::EmptyKernel<void> ))); }namespace cub{

template<> __specialization_static void __wrapper__device_stub_EmptyKernel<void>(void){__device_stub__ZN3cub11EmptyKernelIvEEvv();}}
static void __nv_cudaEntityRegisterCallback( void **__T61) {  __nv_dummy_param_ref(__T61); __nv_save_fatbinhandle_for_managed_rt(__T61); __cudaRegisterEntry(__T61, ((void ( *)(void))cub::EmptyKernel<void> ), _ZN3cub11EmptyKernelIvEEvv, (-1)); __cudaRegisterEntry(__T61, ((void ( *)(float *, float *, float *, int *, float *, float *, int, int, int, float *, float *))kmeans_kernel), _Z13kmeans_kernelPfS_S_PiS_S_iiiS_S_, (-1)); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust6system6detail10sequential3seqE,::thrust::system::detail::sequential::seq), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust8cuda_cub3parE,::thrust::cuda_cub::par), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust8cuda_cub10par_nosyncE,::thrust::cuda_cub::par_nosync), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust12placeholders2_1E,::thrust::placeholders::_1), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust12placeholders2_2E,::thrust::placeholders::_2), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust12placeholders2_3E,::thrust::placeholders::_3), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust12placeholders2_4E,::thrust::placeholders::_4), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust12placeholders2_5E,::thrust::placeholders::_5), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust12placeholders2_6E,::thrust::placeholders::_6), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust12placeholders2_7E,::thrust::placeholders::_7), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust12placeholders2_8E,::thrust::placeholders::_8), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust12placeholders2_9E,::thrust::placeholders::_9), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust12placeholders3_10E,::thrust::placeholders::_10), 0, 1UL, 0, 0); __cudaRegisterVariable(__T61, __shadow_var(_ZN54_INTERNAL_57fa9963_23_wegihted_kmeans_cuda_cu_bb6621726thrust3seqE,::thrust::seq), 0, 1UL, 0, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
