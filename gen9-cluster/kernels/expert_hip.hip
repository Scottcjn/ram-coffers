/* Optional HIP/ROCm expert kernel, for nodes with a working gfx1013 stack.
 *
 * READ THIS BEFORE ENABLING IT.
 *
 * ROCm on console silicon is real but partial. Debian's ROCm packages do build
 * and run against gfx1013 (the PS5's Oberon GPU and the BC-250's Cyan
 * Skillfish), and HIP itself — the compiler, the runtime, the kernel launch
 * path this file uses — works. What does not uniformly work is the *library*
 * layer above it: several ROCm libraries either omit gfx1013 from their target
 * list entirely or fall back to a generic, much slower path, because their
 * build scripts assume a capability level the target does not advertise. Tuned
 * BLAS kernels are the usual casualty.
 *
 * This kernel is therefore written to depend on **nothing but HIP**. No
 * rocBLAS, no hipBLASLt, no Tensile, no library whose gfx1013 support is a
 * question. A GEMV is a hundred lines; a dependency that may silently
 * decay to a generic path is not worth the twenty lines it would save.
 *
 * Consequently the planner never assumes a throughput number for a ROCm node.
 * A node declares `backend: "rocm"` in the inventory *and* a
 * `measured_gemv_gflops` from `g9-probe`, and the planner uses the measurement.
 * A ROCm node with no measurement is planned as if it were Vulkan, which is the
 * conservative direction to be wrong in.
 *
 * Build:
 *     hipcc -O3 --offload-arch=gfx1013 -fPIC -shared expert_hip.hip \
 *           -o libgen9_hip.so
 *
 * If hipcc rejects gfx1013, the installed ROCm predates its support and the
 * Vulkan path is the answer on that node — that is a supported configuration,
 * not a failure.
 */

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define GEN9_FP8_BLOCK 128
#define GEN9_WAVE 32            /* RDNA wave32 */

__device__ __forceinline__ float decode_e4m3(unsigned int b)
{
    const unsigned int sign = (b >> 7) & 0x1u;
    const unsigned int exp = (b >> 3) & 0xFu;
    const unsigned int mant = b & 0x7u;
    float mag;
    if (exp == 0u) {
        mag = float(mant) * 0.125f * exp2f(-6.0f);
    } else {
        mag = (1.0f + float(mant) * 0.125f) * exp2f(float(exp) - 7.0f);
    }
    return sign ? -mag : mag;
}

/* One block per output row; the block reduces in shared memory.
 *
 * A row of a DeepSeek expert is `hidden` wide (7168 on V3, 8192 on the assumed
 * V4-Pro profile), so a 256-thread block gives each thread ~32 weights: enough
 * to keep the loads coalesced and short enough that the reduction is not the
 * dominant cost. */
template <int BLOCK>
__global__ void gemv_fp8_kernel(const uint8_t *__restrict__ w,
                                const float *__restrict__ scales,
                                const float *__restrict__ x,
                                float *__restrict__ y, int rows, int cols,
                                int scales_per_row)
{
    __shared__ float partial[BLOCK];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const uint8_t *row_w = w + (size_t)row * cols;
    const float *row_s = scales + (size_t)row * scales_per_row;

    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BLOCK) {
        sum += decode_e4m3(row_w[col]) * row_s[col / GEN9_FP8_BLOCK] * x[col];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = partial[0];
    }
}

template <int BLOCK>
__global__ void gemv_f32_kernel(const float *__restrict__ w,
                                const float *__restrict__ x,
                                float *__restrict__ y, int rows, int cols)
{
    __shared__ float partial[BLOCK];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float *row_w = w + (size_t)row * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BLOCK) {
        sum += row_w[col] * x[col];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = partial[0];
    }
}

__global__ void silu_mul_kernel(float *__restrict__ gate,
                                const float *__restrict__ up, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = gate[i];
        gate[i] = (v / (1.0f + __expf(-v))) * up[i];
    }
}

extern "C" {

hipError_t gen9_hip_gemv_f32(const float *w, const float *x, float *y,
                             int rows, int cols, hipStream_t stream)
{
    hipLaunchKernelGGL((gemv_f32_kernel<256>), dim3(rows), dim3(256), 0, stream,
                       w, x, y, rows, cols);
    return hipGetLastError();
}

hipError_t gen9_hip_gemv_fp8(const uint8_t *w, const float *scales,
                             const float *x, float *y, int rows, int cols,
                             hipStream_t stream)
{
    const int scales_per_row = cols / GEN9_FP8_BLOCK;
    hipLaunchKernelGGL((gemv_fp8_kernel<256>), dim3(rows), dim3(256), 0, stream,
                       w, scales, x, y, rows, cols, scales_per_row);
    return hipGetLastError();
}

hipError_t gen9_hip_silu_mul(float *gate, const float *up, int n,
                             hipStream_t stream)
{
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(silu_mul_kernel, dim3(blocks), dim3(threads), 0, stream,
                       gate, up, n);
    return hipGetLastError();
}

/* Reports what the runtime actually found, so `g9-probe` can record the gfx
 * target rather than trusting the inventory file's claim about it. */
hipError_t gen9_hip_target(char *out, int len)
{
    hipDeviceProp_t props;
    const hipError_t err = hipGetDeviceProperties(&props, 0);
    if (err != hipSuccess) {
        return err;
    }
    snprintf(out, len, "%s cu=%d clock=%dMHz", props.gcnArchName,
             props.multiProcessorCount, props.clockRate / 1000);
    return hipSuccess;
}

}  /* extern "C" */
