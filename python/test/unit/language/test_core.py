# ruff: noqa: F821,F841
import contextlib
import itertools
import re
from typing import Optional
import math
import textwrap
import pathlib

import numpy as np
import pytest
import torch
import inspect
from numpy.random import RandomState

import triton
import triton.language as tl

from triton._internal_testing import (
    integral_dtypes,
    int_dtypes,
    str_to_triton_dtype,
    uint_dtypes,
    float_dtypes,
    float_dtypes_with_bfloat16,
    dtypes,
    dtypes_with_bfloat16,
    is_cuda,
    is_interpreter,
    is_hopper,
    is_hip,
    is_hip_cdna,
    is_hip_cdna2,
    is_hip_cdna3,
    is_hip_cdna4,
    get_arch,
    torch_float8_dtypes,
    torch_dtypes,
    numpy_random,
    to_triton,
    torch_dtype_name,
    to_numpy,
)


@contextlib.contextmanager
def promotion_numpy_2_0():
    state = np._get_promotion_state()
    np._set_promotion_state("weak")
    try:
        yield
    finally:
        np._set_promotion_state(state)


# No need to emulate NumPy 2.0 if the user has NumPy 2.0
if np.__version__[0] != "1":
    promotion_numpy_2_0 = contextlib.nullcontext

# TODO: enable multiple cta cluster testing.
# num_ctas_list = [1, 4] if torch.cuda.get_device_capability()[0] == 9 else [1]
num_ctas_list = [1]

mma_nonk_sizes = []

GPU_DIALECT = "ttg"
if is_interpreter():
    THREADS_PER_WARP = 1
elif is_hip():
    THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size
    # for CDNA multiple variants of mma instructions are supported:
    # mfma 16x16/mfma 32x32
    # 0 is a special value for automatic heuristic
    if is_hip_cdna():
        mma_nonk_sizes = [0, 16, 32]
else:
    THREADS_PER_WARP = 32


def _bitwidth(dtype: str) -> int:
    # ex.: "int64" -> 64
    return int(re.search(r'(\d+)$', dtype).group(1))


def _dtype(dtype: str) -> str:
    # ex.: "int64" -> "int"
    return re.match(r'([a-zA-Z]+)', dtype).group(0)


def patch_kernel(template, to_replace):
    if is_interpreter():
        local_namespace = {}
        src = textwrap.dedent(inspect.getsource(template.fn))
        for k, v in to_replace.items():
            src = src.replace(k, v)
        exec(src, globals(), local_namespace)
        return local_namespace[template.fn.__name__]
    else:
        kernel = triton.JITFunction(template.fn)
        for key, value in to_replace.items():
            kernel._unsafe_update_src(kernel.src.replace(key, value))
        return kernel


def check_cuda_or_hip(device):
    # CUDA and HIP both use pytorch device 'cuda'.  Other backends like Intel
    # GPU do not.
    if device not in ['cuda']:
        pytest.skip("Only for cuda or HIP")


def check_type_supported(dtype, device):
    '''
    skip test if dtype is not supported on the current device
    '''
    if device in ['cuda']:
        cc = torch.cuda.get_device_capability()
        if cc[0] < 8 and (dtype is tl.bfloat16 or dtype == "bfloat16" or dtype is torch.bfloat16):
            pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")
        if cc[0] < 9 and dtype in {tl.float8e4nv, "float8e4nv", "float8_e4m3fn"}:
            pytest.skip("float8e4nv is only supported on NVGPU with cc >= 90")
    if is_interpreter():
        if dtype in [tl.bfloat16, "bfloat16", torch.bfloat16]:
            pytest.skip("bfloat16 is not supported in the interpreter")


class MfmaLayout:

    def __init__(self, version, warps_per_cta, instr_shape, is_transposed):
        self.version = version
        self.warps_per_cta = warps_per_cta
        self.instr_shape = instr_shape
        self.is_transposed = is_transposed

    def __str__(self):
        return f"#{GPU_DIALECT}.amd_mfma<{{versionMajor={self.version[0]}, versionMinor={self.version[1]}, warpsPerCTA = {self.warps_per_cta}, instrShape={self.instr_shape}, isTransposed = {str(self.is_transposed).lower()}}}>"


class WmmaLayout:

    def __init__(self, version, warps_per_cta):
        self.version = version
        self.warps_per_cta = warps_per_cta

    def __str__(self):
        return f"#{GPU_DIALECT}.amd_wmma<{{version = {self.version}, warpsPerCTA = {self.warps_per_cta}}}>"


class MmaLayout:

    def __init__(self, version, warps_per_cta, ctas_per_cga, cta_split_num, cta_order, instr_shape):
        self.version = version
        self.warps_per_cta = warps_per_cta
        self.ctas_per_cga = ctas_per_cga
        self.cta_split_num = cta_split_num
        self.cta_order = cta_order
        self.instr_shape = instr_shape

    def __str__(self):
        return f"#{GPU_DIALECT}.nvidia_mma<{{versionMajor={self.version[0]}, versionMinor={self.version[1]}, warpsPerCTA={self.warps_per_cta}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}, instrShape={self.instr_shape}}}>"


class DotOperandLayout:

    def __init__(self, parent, op_idx, k_width):
        self.parent = parent
        self.op_idx = op_idx
        self.k_width = k_width

    def __str__(self):
        return f"#{GPU_DIALECT}.dot_op<{{parent={self.parent}, opIdx={self.op_idx}, kWidth={self.k_width}}}>"


class SliceLayout:

    def __init__(self, dim, parent):
        self.dim = dim
        self.parent = parent

    def __str__(self):
        return f"#{GPU_DIALECT}.slice<{{dim = {self.dim}, parent = {self.parent}}}>"


class BlockedLayout:

    def __init__(self, size_per_thread, threads_per_warp, warps_per_cta, order, ctas_per_cga=[1, 1],
                 cta_split_num=[1, 1], cta_order=[0, 1]):
        self.sz_per_thread = size_per_thread
        self.threads_per_warp = threads_per_warp
        self.warps_per_cta = warps_per_cta
        self.order = order
        self.ctas_per_cga = ctas_per_cga
        self.cta_split_num = cta_split_num
        self.cta_order = cta_order

    def __str__(self):
        return f"#{GPU_DIALECT}.blocked<{{sizePerThread={self.sz_per_thread}, threadsPerWarp={self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


class SharedLayout:

    def __init__(self, vec, per_phase, max_phase, order, ctas_per_cga, cta_split_num, cta_order):
        self.vec = vec
        self.per_phase = per_phase
        self.max_phase = max_phase
        self.order = order
        self.ctas_per_cga = ctas_per_cga
        self.cta_split_num = cta_split_num
        self.cta_order = cta_order

    def __str__(self):
        return f"#{GPU_DIALECT}.swizzled_shared<{{vec={self.vec}, perPhase={self.per_phase}, maxPhase={self.max_phase}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


class NVMMASharedLayout:

    def __init__(self, swizzle, transpose, element_bit_width, ctas_per_cga, cta_split_num, cta_order):
        self.swizzle = swizzle
        self.transpose = transpose
        self.element_bit_width = element_bit_width
        self.ctas_per_cga = ctas_per_cga
        self.cta_split_num = cta_split_num
        self.cta_order = cta_order

    def __str__(self):
        transpose_str = "true" if self.transpose else "false"
        return f"#{GPU_DIALECT}.nvmma_shared<{{swizzlingByteWidth={self.swizzle}, transposed={transpose_str}, elementBitWidth={self.element_bit_width}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


class LinearLayout:

    def __init__(self, register, lane, warp, block):
        self.register = register
        self.lane = lane
        self.warp = warp
        self.block = block

    def __str__(self):
        return f"#{GPU_DIALECT}.linear<{{register={self.register}, lane={self.lane}, warp={self.warp}, block={self.block}}}>"


# Python impl of LinearEncodingAttr::basesPerDim
def bases_per_dim(layout, dim, rank, skip_broadcast=True):
    assert isinstance(layout, LinearLayout)
    bases = getattr(layout, dim)
    result = [1] * rank

    if not bases:
        return result

    non_zero_idx = None

    for basis in bases:
        # Find the first non-zero index in the current basis
        idx = next((i for i, v in enumerate(basis) if v != 0), None)
        if idx is not None:
            non_zero_idx = idx
            result[idx] *= 2
        elif not skip_broadcast:
            # If no non-zero found and we're not skipping broadcasts, use the last found non-zero index
            assert non_zero_idx is not None
            result[non_zero_idx] *= 2

    return result


def warps_per_cta(layout, shape):
    if isinstance(layout, LinearLayout):
        return bases_per_dim(layout, 'warp', len(shape))
    elif isinstance(layout, (SliceLayout, DotOperandLayout)):
        return warps_per_cta(layout.parent, shape)
    else:
        return layout.warps_per_cta


def is_layout_applicable(layout) -> bool:
    if isinstance(layout, (BlockedLayout, SharedLayout, LinearLayout)):
        return True
    elif isinstance(layout, SliceLayout):
        return is_layout_applicable(layout.parent)
    elif is_cuda():
        mma_layout = layout.parent if isinstance(layout, DotOperandLayout) else layout
        if not isinstance(mma_layout, MmaLayout):
            return False
        if mma_layout.version[0] >= 3 and not is_hopper():
            return False
        return True
    elif is_hip():
        target_arch = triton.runtime.driver.active.get_current_target().arch
        if "gfx11" in target_arch:
            # RDNA 3
            return isinstance(layout, WmmaLayout)
        elif any(arch for arch in ["gfx8", "gfx9"] if arch in target_arch):
            # CDNA 1, 2, 3
            return isinstance(layout, MfmaLayout)
        else:
            return False
    else:
        return True


def filter_layouts(layouts):
    return [l for l in layouts if is_layout_applicable(l)]


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x", list(dtypes) + ["bfloat16"])
def test_empty_kernel(dtype_x, device):
    SIZE = 128

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        pass

    check_type_supported(dtype_x, device)
    x = to_triton(numpy_random(SIZE, dtype_str=dtype_x), device=device, dst_type=dtype_x)
    kernel[(1, )](x, SIZE=SIZE, num_warps=4)


def test_scalar_overflow(device):

    @triton.jit
    def kernel():
        huge_int: tl.constexpr = 0xFFFFFFFFFFFFFF
        x = tl.full((), 32, dtype=tl.int32)
        y = x + huge_int

    with pytest.raises(triton.TritonError, match="out of range"):
        kernel[(1, )]()


# generic test functions
def _test_unary(dtype_x, expr, numpy_expr=None, device='cuda', num_ctas=1):
    check_type_supported(dtype_x, device)  # early return if dtype_x is not supported
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    # inputs
    x = numpy_random(SIZE, dtype_str=dtype_x)
    if 'log' in expr:
        x = np.abs(x) + 0.01
    # reference result
    z_ref = eval(expr if numpy_expr is None else numpy_expr)
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    z_tri = to_triton(np.empty_like(x), device=device, dst_type=dtype_x)
    kernel[(1, )](Z=z_tri, X=x_tri, SIZE=SIZE, num_warps=4, num_ctas=num_ctas)
    # compare
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)


def _binary_op_dtype_override(a: str, b: str) -> Optional[np.dtype]:
    """
    Given two dtype strings, returns the numpy dtype Triton thinks binary
    operations on the two types should return. Returns None if the return value
    matches numpy. This is generally needed because Triton and pytorch return
    narrower floating point types than numpy in mixed operations, and because
    Triton follows C/C++ semantics around mixed signed/unsigned operations, and
    numpy/pytorch do not.
    """
    overrides = {
        ('float16', 'int16'): np.float16,
        ('float16', 'int32'): np.float16,
        ('float16', 'int64'): np.float16,
        ('float16', 'uint16'): np.float16,
        ('float16', 'uint32'): np.float16,
        ('float16', 'uint64'): np.float16,
        ('int8', 'uint8'): np.uint8,
        ('int8', 'uint16'): np.uint16,
        ('int8', 'uint32'): np.uint32,
        ('int8', 'uint64'): np.uint64,
        ('int16', 'uint16'): np.uint16,
        ('int16', 'uint32'): np.uint32,
        ('int16', 'uint64'): np.uint64,
        ('int32', 'uint32'): np.uint32,
        ('int32', 'uint64'): np.uint64,
        ('int64', 'uint64'): np.uint64,
    }
    key = (a, b) if a < b else (b, a)
    return overrides.get(key)


def _test_binary(dtype_x, dtype_y, expr, numpy_expr=None, mode_x='real', mode_y='real', device='cuda', num_ctas=1,
                 x_low=None, x_high=None, y_low=None, y_high=None, filter_y=None, test_broadcast=True,
                 test_scalar=True):
    check_type_supported(dtype_x, device)  # early return if dtype_x is not supported
    check_type_supported(dtype_y, device)
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    @triton.jit
    def kernel_broadcast_lhs(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    @triton.jit
    def kernel_broadcast_rhs(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    @triton.jit
    def kernel_scalar_rhs(Z, X, y: tl.constexpr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    replacements = {'GENERATE_TEST_HERE': expr}
    kernel = patch_kernel(kernel, replacements)
    kernel_broadcast_lhs = patch_kernel(kernel_broadcast_lhs, replacements)
    kernel_broadcast_rhs = patch_kernel(kernel_broadcast_rhs, replacements)
    kernel_scalar_rhs = patch_kernel(kernel_scalar_rhs, replacements)

    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs, low=x_low, high=x_high)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs, low=y_low, high=y_high)
    if filter_y:
        y[filter_y(y)] = 1
    if mode_x == 'nan':
        x[:] = float('nan')
    if mode_y == 'nan':
        y[:] = float('nan')

    def do_test(x, y, kernel_fn):
        x_is_scalar = isinstance(x, (bool, int, float))
        y_is_scalar = isinstance(y, (bool, int, float))
        scalar_test = x_is_scalar or y_is_scalar

        # For scalars, we follow the NumPy 2.0 (and JAX/PyTorch pretty much) casting rules.
        if scalar_test:
            # We remove any explicit casting
            pattern = r'\.astype\(np\.\w+\)'
            scalar_expr = expr if numpy_expr is None else re.sub(pattern, '', numpy_expr)
            with promotion_numpy_2_0():
                z_ref = eval(scalar_expr)
        else:
            z_ref = eval(expr if numpy_expr is None else numpy_expr)

        dtype_z = _binary_op_dtype_override(dtype_x, dtype_y)
        if not scalar_test and dtype_z is not None:
            z_ref = z_ref.astype(dtype_z)

        # triton result
        x_tri = x if x_is_scalar else to_triton(x, device=device, dst_type=dtype_x)
        y_tri = y if y_is_scalar else to_triton(y, device=device, dst_type=dtype_y)
        z_tri = to_triton(np.empty(SIZE, dtype=z_ref.dtype), device=device)
        kernel_fn[(1, )](z_tri, x_tri, y_tri, SIZE=SIZE, num_warps=4, num_ctas=num_ctas)
        err_msg = f"{expr}, {kernel_fn.__name__}"
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), err_msg=err_msg, atol=7e-3, rtol=0.01)

    def get_scalar(x, dtype, low, high, filter):
        # If dtype is int, don't choose a huge number for the scalar
        # as it'll overflow easily when converted to the other dtype
        if dtype in integral_dtypes:
            # Choose in range [-7, 7] ([0, 7] for uints)
            low_x = 0 if dtype in uint_dtypes else -7
            if low is not None:
                low_x = max(low_x, low)
            high_x = 7
            if high is not None:
                high_x = min(high_x, high)
            scalar = numpy_random((), dtype_str=dtype, rs=rs, low=low_x, high=high_x).item()
            if filter and filter(scalar):
                #  https://xkcd.com/221/
                scalar = 4
        else:
            scalar = x.flat[0].item()
        return scalar

    do_test(x, y, kernel)
    if mode_y != 'nan' and test_scalar:
        if dtype_x in uint_dtypes:
            low = 0 if y_low is None else max(y_low, 0)
        else:
            low = y_low
        y_scalar = get_scalar(y, dtype_y, low, y_high, filter_y)
        do_test(x, y_scalar, kernel_scalar_rhs)
    if test_broadcast:
        do_test(x[:1].reshape(()), y, kernel_broadcast_lhs)
        do_test(x, y[:1].reshape(()), kernel_broadcast_rhs)


def _min_max_integral_mod_value(dtype_x, dtype_y) -> Optional[int]:
    """
    Limit min/max values for integral types for mod values. Leads to
    overflow/underflow when casting large integral types to floats.
    """
    x_bitwidth = _bitwidth(dtype_x)
    y_bitwidth = _bitwidth(dtype_y)

    # hard cap max value bit-width to 32 if 64 bit-width types
    min_bitwidth = min(x_bitwidth, y_bitwidth, 32)

    # Limit max value bit-width to be one integral type less than the min bit-width
    # For example:
    #   int64, float32 -> int16
    #   uint16, float16 -> uint8
    x_dtype = _dtype(dtype_x)
    max_bitwidth = max(min_bitwidth >> 1, 8)
    dtype_max = x_dtype + str(max_bitwidth)

    max_info = np.iinfo(getattr(np, dtype_max))

    # Still need to limit values here for uints
    if max_bitwidth >= 16 and dtype_max in uint_dtypes:
        return max_info.min, max_info.max // 4
    else:
        return max_info.min, max_info.max


def test_dtype_codegen():
    for dtype in dtypes_with_bfloat16:
        full_name = f"triton.language.{dtype}"
        assert repr(eval(full_name)) == full_name


# ---------------
# test binary ops
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, dtype_y, op", [  #
    (dtype_x, dtype_y, op)
    for op in ['+', '-', '*', '/', '%']
    for dtype_x in dtypes_with_bfloat16
    for dtype_y in dtypes_with_bfloat16
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_bin_op(dtype_x, dtype_y, op, num_ctas, device):
    expr = f'x {op} y'
    np_expr_gen = (lambda x, y: f'{x} {op} {y}') if op != '%' else (lambda x, y: f'np.fmod({x}, {y})')

    # Triton promotes 16-bit floating-point / and % to 32-bit because there
    # are no native div or FRem operations on float16. Since we have to
    # convert anyway, we may as well take the accuracy bump.
    def promote_to_fp32(dtype_x, dtype_y):
        return dtype_x in ('float16', 'bfloat16') and dtype_y not in ('float32', 'float64')

    if op in ('/', '%') and (promote_to_fp32(dtype_x, dtype_y) or promote_to_fp32(dtype_y, dtype_x)):
        numpy_expr = np_expr_gen('x.astype(np.float32)', 'y.astype(np.float32)')
    elif (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = np_expr_gen(f'x.astype(np.{dtype_x})', f'y.astype(np.{dtype_x})')
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = np_expr_gen(f'x.astype(np.{dtype_y})', f'y.astype(np.{dtype_y})')
    elif op == '%':
        # LLVM has 'numpy.fmod', not 'numpy.remainder', semantics on integer remainders.
        numpy_expr = np_expr_gen('x', 'y')
    else:
        numpy_expr = None

    if (op in ('%', '/') and ((dtype_x in int_dtypes and dtype_y in uint_dtypes) or
                              (dtype_x in uint_dtypes and dtype_y in int_dtypes))):
        with pytest.raises(triton.TritonError, match='Cannot use .* because they have different signedness'):
            _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas)
    else:
        # skip when bfloat16, as NumPy's ref performs the computation in float32
        # while Triton performs it in bfloat16
        skip_scalar_test = ((dtype_x == "bfloat16" and "float" in dtype_y)
                            or (op in ('/', '%') and dtype_x in ("float16", "bfloat16")))
        # can't divide by zero
        not_zero = op in ('/', '%') and dtype_x in integral_dtypes and dtype_y in integral_dtypes
        # can't represent -int(max)
        not_minus_one = op in ('*', '/') and dtype_x in int_dtypes and dtype_y in int_dtypes
        if not_zero or not_minus_one:
            filter_y = lambda y: not_zero * (y == 0) | not_minus_one * (y == -1)
        else:
            filter_y = None

        if op == "%" and dtype_x in integral_dtypes and dtype_y in float_dtypes_with_bfloat16:
            x_low, x_high = _min_max_integral_mod_value(dtype_x, dtype_y)
        else:
            x_low, x_high = None, None

        _test_binary(
            dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas,
            # fails with values where fmod(x, y) is roughly zero, but happens to
            # pass with the random values chosen for non-broadcast tests
            test_broadcast=(op != "%"), x_low=x_low, x_high=x_high, filter_y=filter_y, test_scalar=not skip_scalar_test)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype, order", [(dtype, order) for dtype in dtypes_with_bfloat16 for order in [0, 1]])
def test_addptr(dtype, order, device):
    check_type_supported(dtype, device)

    @triton.jit
    def kernel(x, y, ORDER: tl.constexpr, SIZE: tl.constexpr):
        offs = tl.arange(0, SIZE)
        if ORDER == 0:
            tl.store(y + offs, tl.load(x + offs))
        else:
            tl.store(offs + y, tl.load(offs + x))

    SIZE = 1024
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    x_tri = to_triton(x, dst_type=dtype, device=device)
    y_tri = to_triton(y, dst_type=dtype, device=device)
    y = x
    kernel[
        1,
    ](x_tri, y_tri, order, SIZE)
    np.testing.assert_allclose(y, to_numpy(y_tri))


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, dtype_y", [  #
    (dtype_x, dtype_y) for dtype_x in int_dtypes for dtype_y in int_dtypes
] + [(dtype_x, dtype_y) for dtype_x in uint_dtypes for dtype_y in uint_dtypes])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_floordiv(dtype_x, dtype_y, num_ctas, device):
    # Triton has IEEE, not numpy/torch, semantics for %, and those carry
    # through to //, so we have to use a nonstandard expression to get a
    # reference result for //.
    expr = 'x // y'
    numpy_expr = '((x - np.fmod(x, y)) / y)'
    # can't represent -int(max)
    not_minus_one = dtype_x in int_dtypes and dtype_y in int_dtypes
    if not_minus_one:
        filter_y = lambda y: y == -1
    else:
        filter_y = None
    _test_binary(dtype_x, dtype_y, expr, numpy_expr, filter_y=filter_y, device=device, num_ctas=num_ctas)


def test_unsigned_name_mangling(device):
    # Test that uint32 and int32 are mangled differently by the compiler
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(O1, O2, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        out1 = tl.abs(x)  # uint32 -> nop
        out2 = tl.abs(-y)  # int32 -> should have an effect
        tl.store(O1 + off, out1)
        tl.store(O2 + off, out2)

    dtype_x = 'uint32'
    dtype_y = 'int32'
    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs)
    # reference result
    expect = (np.abs(x), np.abs(-y))
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    y_tri = to_triton(y, device=device, dst_type=dtype_y)
    actual = tuple(to_triton(np.empty_like(e), device=device) for e in expect)
    kernel[(1, )](actual[0], actual[1], x_tri, y_tri, SIZE=SIZE, num_warps=4)

    # Bitwise op, so expect exact equality
    assert (expect[0] == to_numpy(actual[0])).all()
    assert (expect[1] == to_numpy(actual[1])).all()


# test bitwise ops
# ---------------
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, dtype_y, op", [  #
    (dtype_x, dtype_y, op)
    for op in ['&', '|', '^']
    for dtype_x in dtypes + dtypes_with_bfloat16
    for dtype_y in dtypes + dtypes_with_bfloat16
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_bitwise_op(dtype_x, dtype_y, op, num_ctas, device):
    expr = f'x {op} y'
    if (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    if 'float' in dtype_x + dtype_y:
        # The CompilationError must have been caused by a C++ exception with this text.
        with pytest.raises(triton.TritonError, match='invalid operands of type'):
            _test_binary(dtype_x, dtype_y, expr, numpy_expr='np.array([])', device=device, num_ctas=num_ctas)
    else:
        _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, dtype_y, op", [  #
    (dtype_x, dtype_y, op) for op in ['<<', '>>'] for dtype_x in int_dtypes + uint_dtypes for dtype_y in uint_dtypes
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_shift_op(dtype_x, dtype_y, op, num_ctas, device):
    expr = f'x {op} y'
    bw = max(_bitwidth(dtype_x), _bitwidth(dtype_y))
    if dtype_x.startswith('int'):
        dtype_z = f'int{bw}'
    else:
        dtype_z = f'uint{bw}'
    numpy_expr = f'x.astype(np.{dtype_z}) {op} y.astype(np.{dtype_z})'
    _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas, y_low=0, y_high=bw)


# ---------------
# test compare ops
# ---------------
ops = ['==', '!=', '>', '<', '>=', '<=']


@pytest.mark.interpreter
@pytest.mark.parametrize(
    "dtype_x, dtype_y, op, mode_x, mode_y",
    # real
    [(dtype_x, dtype_y, op, 'real', 'real') for op in ops for dtype_x in dtypes for dtype_y in dtypes]
    # NaNs
    + [('float32', 'float32', op, mode_x, mode_y)
       for op in ops
       for mode_x, mode_y in [('nan', 'real'), ('real', 'nan'), ('nan', 'nan')]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_compare_op(dtype_x, dtype_y, op, mode_x, mode_y, num_ctas, device):
    expr = f'x {op} y'
    if (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    _test_binary(dtype_x, dtype_y, expr, numpy_expr, mode_x=mode_x, mode_y=mode_y, device=device, num_ctas=num_ctas)


# ---------------
# test broadcast
# ---------------
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", dtypes_with_bfloat16)
def test_broadcast(dtype, device):
    check_type_supported(dtype, device)

    @triton.jit
    def broadcast_kernel(x_ptr, y_ptr, y_broadcasted_ptr, M: tl.constexpr, N: tl.constexpr):
        offset1 = tl.arange(0, M)
        offset2 = tl.arange(0, N)
        x = tl.load(x_ptr + N * offset1[:, None] + offset2[None, :])
        y = tl.load(y_ptr + offset2)
        _, y_broadcasted = tl.broadcast(x, y)
        tl.store(y_broadcasted_ptr + N * offset1[:, None] + offset2[None, :], y_broadcasted)

    M = 32
    N = 64
    rs = RandomState(17)
    x = numpy_random((M, N), dtype_str=dtype, rs=rs)
    y = numpy_random(N, dtype_str=dtype, rs=rs)
    _, y_broadcasted_np = np.broadcast_arrays(x, y)

    x_tri = to_triton(x, device=device, dst_type=dtype)
    y_tri = to_triton(y, device=device, dst_type=dtype)
    y_broadcasted_tri = to_triton(np.empty((M, N), dtype=y_broadcasted_np.dtype), device=device, dst_type=dtype)

    broadcast_kernel[(1, )](x_tri, y_tri, y_broadcasted_tri, M=M, N=N)
    assert (y_broadcasted_np == to_numpy(y_broadcasted_tri)).all()


# ----------
# test slice
# ----------


@pytest.mark.interpreter
def test_slice(device):

    @triton.jit
    def slice_kernel(XBLOCK: tl.constexpr):
        data = tl.arange(0, XBLOCK)
        tl.static_assert(data.shape == [XBLOCK])

        t = data[None, :]
        tl.static_assert(t.shape == [1, XBLOCK])

        t = data[None, :, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        scalar = tl.full([], 1, tl.int32)
        tl.static_assert(scalar.shape == [])

        t = scalar[None]
        tl.static_assert(t.shape == [1])

        t = scalar[None, None]
        tl.static_assert(t.shape == [1, 1])

    slice_kernel[(1, )](XBLOCK=32)


# ------------------
# test invalid slice
# ------------------


@pytest.mark.interpreter
def test_invalid_slice(device):
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst):
        dst[10:]

    with pytest.raises(triton.TritonError, match='unsupported tensor index'):
        _kernel[(1, )](dst=dst)


# ----------------
# test expand_dims
# ----------------
@pytest.mark.interpreter
def test_expand_dims(device):

    @triton.jit
    def expand_dims_kernel(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, 0)
        tl.static_assert(t.shape == [1, N])

        t = tl.expand_dims(offset1, 1)
        tl.static_assert(t.shape == [N, 1])

        t = tl.expand_dims(offset1, -1)
        tl.static_assert(t.shape == [N, 1])

        t = tl.expand_dims(offset1, -2)
        tl.static_assert(t.shape == [1, N])

        t = tl.expand_dims(offset1, (0, -1))
        tl.static_assert(t.shape == [1, N, 1])

        t = tl.expand_dims(offset1, (0, 1, 3))
        tl.static_assert(t.shape == [1, 1, N, 1])

        t = tl.expand_dims(offset1, (-4, 2, -1))
        tl.static_assert(t.shape == [1, N, 1, 1])

        t = tl.expand_dims(offset1, (3, 1, 2))
        tl.static_assert(t.shape == [N, 1, 1, 1])

        scalar = tl.sum(offset1)
        tl.static_assert(scalar.shape == [])
        t = tl.expand_dims(scalar, 0)
        tl.static_assert(t.shape == [1])

        t = tl.expand_dims(scalar, -1)
        tl.static_assert(t.shape == [1])

        # N is a scalar that's not even a tl.tensor -- this should work too.
        t = tl.expand_dims(N, -1)
        tl.static_assert(t.shape == [1])

    N = 32
    dummy_tensor = torch.empty((), device=device)
    expand_dims_kernel[(1, )](dummy_tensor, N)


@pytest.mark.interpreter
def test_expand_dims_error_cases(device):

    @triton.jit
    def dim_out_of_range1(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, -2)
        t = tl.expand_dims(offset1, -3)

    @triton.jit
    def dim_out_of_range2(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, 1)
        t = tl.expand_dims(offset1, 2)

    @triton.jit
    def dim_out_of_range3(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, 1)
        scalar = tl.sum(offset1)

        t = tl.expand_dims(scalar, 1)

    @triton.jit
    def duplicate_dim1(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, (0, 0))

    @triton.jit
    def duplicate_dim2(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, (0, -3))

    N = 32
    dummy_tensor = torch.empty((), device=device)

    with pytest.raises(triton.TritonError) as exc_info:
        dim_out_of_range1[(1, )](dummy_tensor, N)
    assert "invalid axis -3" in str(exc_info.value.__cause__)

    with pytest.raises(triton.TritonError) as exc_info:
        dim_out_of_range2[(1, )](dummy_tensor, N)
    assert "invalid axis 2" in str(exc_info.value.__cause__)

    with pytest.raises(triton.TritonError) as exc_info:
        dim_out_of_range3[(1, )](dummy_tensor, N)
    assert "invalid axis 1" in str(exc_info.value.__cause__)

    with pytest.raises(triton.TritonError) as exc_info:
        duplicate_dim1[(1, )](dummy_tensor, N)
    assert re.search(r"duplicate axes, normalized axes = \[0, 0\]", str(exc_info.value.__cause__))

    with pytest.raises(triton.TritonError) as exc_info:
        duplicate_dim2[(1, )](dummy_tensor, N)
    assert re.search(r"duplicate axes, normalized axes = \[0, 0\]", str(exc_info.value.__cause__))


# ----------------------------
# test invalid program id axis
# ----------------------------
@pytest.mark.interpreter
def test_invalid_pid_axis(device):
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst):
        pid = tl.program_id(20)

    with pytest.raises(triton.TritonError) as exc_info:
        _kernel[(1, )](dst)
    assert re.search(r"program_id axis must be 0, 1, or 2 but got 20", str(exc_info.value.__cause__))


# ---------------
# test where
# ---------------
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", dtypes_with_bfloat16 + ["*int32"])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_where(dtype, num_ctas, device):
    select_ptrs = False
    if dtype == "*int32":
        dtype = "int64"
        select_ptrs = True
    check_type_supported(dtype, device)

    @triton.jit
    def where_kernel(cond_ptr, a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
                     TEST_POINTERS: tl.constexpr, TEST_SCALAR_POINTERS: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        decide = tl.load(cond_ptr + offsets, mask=mask)
        if TEST_SCALAR_POINTERS:
            ptr = tl.where(tl.load(cond_ptr), a_ptr, b_ptr)
            output = tl.load(ptr + offsets, mask=mask)
        else:
            if TEST_POINTERS:
                a = tl.load(a_ptr + offsets, mask=mask).to(tl.pi32_t)
                b = tl.load(b_ptr + offsets, mask=mask).to(tl.pi32_t)
            else:
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
            output = tl.where(decide, a, b)
        tl.store(output_ptr + offsets, output, mask=mask)

    SIZE = 1_000
    rs = RandomState(17)
    cond = numpy_random(SIZE, 'bool', rs)
    x = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    z = np.where(cond, x, y)

    cond_tri = to_triton(cond, device=device)
    x_tri = to_triton(x, device=device, dst_type=dtype)
    y_tri = to_triton(y, device=device, dst_type=dtype)
    z_tri = to_triton(np.empty(SIZE, dtype=z.dtype), device=device, dst_type=dtype)

    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']), )
    where_kernel[grid](cond_tri, x_tri, y_tri, z_tri, SIZE, BLOCK_SIZE=1024, TEST_POINTERS=select_ptrs,
                       TEST_SCALAR_POINTERS=False, num_ctas=num_ctas)
    assert (z == to_numpy(z_tri)).all()
    if select_ptrs:
        where_kernel[grid](cond_tri, x_tri, y_tri, z_tri, SIZE, BLOCK_SIZE=1024, TEST_POINTERS=select_ptrs,
                           TEST_SCALAR_POINTERS=True)
        z = np.where(cond[0], x, y)
        assert (z == to_numpy(z_tri)).all()


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_where_broadcast(num_ctas, device):

    @triton.jit
    def where_kernel(cond_ptr, a_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        xoffsets = tl.arange(0, BLOCK_SIZE)[:, None]
        yoffsets = tl.arange(0, BLOCK_SIZE)[None, :]

        mask = tl.load(cond_ptr + yoffsets)
        vals = tl.load(a_ptr + yoffsets + BLOCK_SIZE * xoffsets)
        res = tl.where(mask, vals, 0.)
        tl.store(out_ptr + yoffsets + BLOCK_SIZE * xoffsets, res)

    @triton.jit
    def where_scalar_condition(a_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        xoffsets = tl.arange(0, BLOCK_SIZE)[:, None]
        yoffsets = tl.arange(0, BLOCK_SIZE)[None, :]
        mask = False
        vals = tl.load(a_ptr + yoffsets + BLOCK_SIZE * xoffsets)
        res = tl.where(mask, vals, 0.)
        tl.store(out_ptr + yoffsets + BLOCK_SIZE * xoffsets, res)

    SIZE = 32
    dtype = 'float32'
    rs = RandomState(17)
    x = numpy_random((SIZE, SIZE), dtype_str=dtype, rs=rs)
    mask = numpy_random(SIZE, 'bool', rs=rs)
    z = np.where(mask, x, 0)
    cond_tri = to_triton(mask, device=device)
    x_tri = to_triton(x, device=device, dst_type=dtype)
    z_tri = to_triton(np.empty((SIZE, SIZE), dtype=z.dtype), device=device, dst_type=dtype)
    where_kernel[(1, )](cond_tri, x_tri, z_tri, SIZE)
    assert (z == to_numpy(z_tri)).all()
    where_scalar_condition[(1, )](x_tri, z_tri, SIZE, num_ctas=num_ctas)
    z = np.where(0, x, 0)
    assert (z == to_numpy(z_tri)).all()


# ---------------
# test unary ops
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, expr",
                         [(dtype_x, ' -x') for dtype_x in dtypes_with_bfloat16] + [(dtype_x, ' ~x')
                                                                                   for dtype_x in int_dtypes])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_unary_op(dtype_x, expr, num_ctas, device):
    _test_unary(dtype_x, expr, device=device, num_ctas=num_ctas)


# ----------------
# test math ops
# ----------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, expr, x",
                         [(dtype_x, expr, x)
                          for dtype_x in ["float32", "float64"]
                          for expr in ['exp', 'log', 'cos', 'sin', 'exp2', 'log2', 'sqrt', 'floor', 'ceil']
                          for x in ['x', '3.0']])
def test_math_op(dtype_x, expr, x, device):
    _test_unary(dtype_x, f'tl.{expr}({x})', f'np.{expr}({x}) ', device=device)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", [dtype for dtype in ["float32", "float64"]])
def test_math_erf_op(dtype, device):
    check_type_supported(dtype, device)
    SIZE = 128

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = tl.math.erf(x)
        tl.store(Z + off, z)

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    x = torch.randn(SIZE, dtype=torch_dtype, device=device)
    z_ref = torch.erf(x)
    z_tri = torch.zeros_like(x)
    kernel[(1, )](z_tri, x, SIZE=SIZE, num_warps=4)
    torch.testing.assert_close(z_tri, z_ref)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", [dtype for dtype in ["float32", "float64"]])
def test_math_fma_op(dtype, device):
    check_type_supported(dtype, device)
    SIZE = 128

    @triton.jit
    def kernel(Z, X, Y, W, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        w = tl.load(W + off)
        z = tl.math.fma(x, y, w)
        tl.store(Z + off, z)

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    x = torch.randn(SIZE, dtype=torch_dtype, device=device)
    y = torch.randn(SIZE, dtype=torch_dtype, device=device)
    w = torch.randn(SIZE, dtype=torch_dtype, device=device)
    z_ref = x * y + w
    z_tri = torch.zeros_like(x)
    kernel[(1, )](z_tri, x, y, w, SIZE=SIZE, num_warps=4)
    torch.testing.assert_close(z_tri, z_ref)


@pytest.mark.interpreter
@pytest.mark.parametrize("expr", ["tl.math.fdiv(x, y)", "tl.math.div_rn(x, y)"])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_math_divide_op(expr, num_ctas, device):
    numpy_expr = "x / y"
    dtype = "float32"
    _test_binary(dtype, dtype, expr, numpy_expr, device=device, num_ctas=num_ctas)


# -------------
# test precise math
# -------------
@pytest.mark.interpreter
@pytest.mark.parametrize("expr_prec, expr_ref",
                         [('tl.math.sqrt_rn(x)', 'tl.math.sqrt(x.to(tl.float64)).to(tl.float32)'),
                          ('tl.math.div_rn(x,y)', '(x.to(tl.float64) / y.to(tl.float64)).to(tl.float32)')])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_precise_math(expr_prec, expr_ref, num_ctas, device):

    @triton.jit
    def kernel(X, Y, OUT, OUT_REF, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = tl.load(Y + tl.arange(0, BLOCK))
        prec = PREC_CALC
        ref = REF_CALC
        tl.store(OUT + tl.arange(0, BLOCK), prec)
        tl.store(OUT_REF + tl.arange(0, BLOCK), ref)

    shape = (128, )
    out = torch.zeros(shape, dtype=torch.float32, device=device)
    out_ref = torch.zeros(shape, dtype=torch.float32, device=device)

    x = torch.randn(shape, dtype=torch.float32, device=device)
    y = torch.randn(shape, dtype=torch.float32, device=device)

    if (expr_prec.count('sqrt') > 0):
        x = torch.abs(x)

    if (expr_prec.count('div') > 0):
        y += 1e-6

    kernel = patch_kernel(kernel, {'PREC_CALC': expr_prec, 'REF_CALC': expr_ref})

    kernel[(1, )](x, y, out, out_ref, BLOCK=shape[0], num_ctas=num_ctas)
    assert torch.all(out == out_ref)  # bitwise exact


# ----------------
# test abs
# ----------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x", [(dtype_x) for dtype_x in dtypes_with_bfloat16])
def test_abs(dtype_x, device):
    _test_unary(dtype_x, 'tl.abs(x)', 'np.abs(x) ', device=device)


@pytest.mark.interpreter
@pytest.mark.parametrize("in_dtype", [tl.float8e4b15, tl.float8e4nv, tl.float8e5])
def test_abs_fp8(in_dtype, device):
    if is_hip():
        pytest.skip('test_abs_fp8 not supported on HIP.')
    elif is_cuda():
        cc = torch.cuda.get_device_capability()
        if in_dtype == tl.float8e4b15 and cc >= (9, 0):
            pytest.skip("float8e4b15 not supported on CUDA >= 9.0")
        if in_dtype == tl.float8e4nv and cc < (8, 9):
            pytest.skip("float8e4nv not supported on CUDA < 8.9")

    @triton.jit
    def abs_kernel(X, Z, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = tl.abs(x)
        tl.store(Z + off, z)

    f8_tensor = torch.tensor(range(-128, 128), dtype=torch.int8, device=device)
    # f32_to_f8 doesn't handle nan, so we make sure f8_tensor doesn't contain any nan
    all_exp_ones = (f8_tensor & 0b01111100) == 128 - 2**in_dtype.fp_mantissa_width
    f8_tensor[all_exp_ones] = 0
    f8 = triton.reinterpret(f8_tensor, in_dtype)
    n_elements = f8_tensor.numel()
    out_f8 = torch.empty_like(f8_tensor)
    abs_kernel[(1, )](f8, triton.reinterpret(out_f8, in_dtype), n_elements)

    f32_tensor = convert_float_to_float32(f8_tensor, in_dtype)
    expect = f32_tensor.abs()
    actual_f8 = convert_float_to_float32(out_f8, in_dtype)
    torch.testing.assert_close(actual_f8, expect, equal_nan=True)


# ----------------
# test passing shapes as individual params rather than tuples
# ----------------


@pytest.mark.interpreter
def test_shapes_as_params(device):

    @triton.jit
    def kernel():
        a = tl.arange(0, 32).expand_dims(-1).broadcast_to(32, 32)
        tl.static_assert(a.shape == [tl.constexpr(32), tl.constexpr(32)])

        a = tl.arange(0, 32).reshape(4, 8).permute(1, 0)
        tl.static_assert(a.shape == [tl.constexpr(8), tl.constexpr(4)])

        a = tl.arange(0, 32).reshape(4, 8).trans()
        tl.static_assert(a.shape == [tl.constexpr(8), tl.constexpr(4)])

        a = tl.arange(0, 32).reshape(4, 8).reshape(32)
        tl.static_assert(a.shape == [tl.constexpr(32)])

        a = tl.arange(0, 64).reshape(2, 4, 8).trans(2, 1, 0)
        tl.static_assert(a.shape == [tl.constexpr(8), tl.constexpr(4), tl.constexpr(2)])

        a = tl.arange(0, 64).reshape(2, 4, 8).trans((2, 1, 0))
        tl.static_assert(a.shape == [tl.constexpr(8), tl.constexpr(4), tl.constexpr(2)])

        a = tl.arange(0, 64).view(2, 4, 8)
        tl.static_assert(a.shape == [tl.constexpr(2), tl.constexpr(4), tl.constexpr(8)])

    kernel[(1, )]()


# ----------------
# test transpose
# ----------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x", [(dtype_x) for dtype_x in dtypes_with_bfloat16])
def test_transpose(dtype_x, device):
    check_type_supported(dtype_x, device)
    SIZE = 128

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        off2d = off[None, :] + (tl.arange(0, 2) * SIZE)[:, None]
        x = tl.load(X + off2d)
        z = x.T
        tl.store(Z + off2d.T, z)

    x = numpy_random([SIZE, 2], dtype_str=dtype_x)
    z_ref = x.T
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    z_tri = to_triton(np.empty_like(z_ref), device=device, dst_type=dtype_x)
    kernel[(1, )](z_tri, x_tri, SIZE=SIZE)
    np.testing.assert_allclose(z_ref, to_numpy(z_tri))


# ----------------
# test indexing
# ----------------


def make_ptr_str(name, shape):
    rank = len(shape)
    offsets = []
    stride = 1
    for i in reversed(range(rank)):
        idx = ', '.join([':' if ii == i else 'None' for ii in range(rank)])
        offsets += [f'tl.arange(0, {shape[i]})[{idx}]*{stride}']
        stride *= shape[i]
    return f"{name} + {' + '.join(offsets)}"


# TODO: handle `%4 = ttg.convert_layout %3 : tensor<32xi32, #blocked0> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>``
@pytest.mark.parametrize("expr, dtype_str", [(f'x[{s}]', d)
                                             for s in ['None, :', ':, None', 'None, :, :', ':, :, None']
                                             for d in ['int32', 'uint32', 'uint16']])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_index1d(expr, dtype_str, num_ctas, device):
    rank_x = expr.count(':')
    rank_y = expr.count(',') + 1
    shape_x = [32 for _ in range(rank_x)]
    shape_z = [32 for _ in range(rank_y)]
    shape_z_rank_mismatch = [32 for _ in range(rank_y + 1)]
    shape_z_dim_mismatch = [64 for _ in range(rank_y)]

    # Triton kernel
    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        m = tl.arange(0, SIZE)
        n = tl.arange(0, SIZE)
        x = tl.load(X_PTR_EXPR)
        z = GENERATE_TEST_HERE
        tl.store(Z_PTR_EXPR, z)

    def generate_kernel(shape_x, shape_z):
        to_replace = {
            'X_PTR_EXPR': make_ptr_str('X', shape_x),
            'Z_PTR_EXPR': make_ptr_str('Z', shape_z),
            'GENERATE_TEST_HERE': expr,
        }
        return patch_kernel(kernel, to_replace)

    kernel_match = generate_kernel(shape_x, shape_z)
    kernel_dim_mismatch = generate_kernel(shape_x, shape_z_dim_mismatch)
    kernel_rank_mismatch = generate_kernel(shape_x, shape_z_rank_mismatch)

    # torch result
    x = numpy_random(shape_x, dtype_str=dtype_str)
    y = np.zeros(shape_z, dtype=getattr(np, dtype_str))
    z_ref = eval(expr) + y
    # triton result
    z_tri = to_triton(np.empty_like(z_ref), device=device)
    x_tri = to_triton(x, device=device)
    kernel_match[(1, )](z_tri, x_tri, num_warps=1, SIZE=shape_x[0])
    # compare
    assert (z_ref == to_numpy(z_tri)).all()

    def catch_compilation_error(kernel):
        try:
            kernel[(1, )](z_tri, x_tri, num_warps=1, SIZE=shape_x[0], num_ctas=num_ctas)
        except triton.CompilationError as e:
            np.testing.assert_(True)
        except BaseException:
            np.testing.assert_(False)

    catch_compilation_error(kernel_dim_mismatch)
    catch_compilation_error(kernel_rank_mismatch)


# ---------------
# test tuples
# ---------------


@triton.jit
def tuples_fn(a, b):
    return a + b, \
        a - b, \
        a * b


@pytest.mark.interpreter
def test_tuples(device):

    @triton.jit
    def with_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = tuples_fn(x, y)
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    @triton.jit
    def without_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = x + y, x - y, x * y
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    x = torch.tensor([1.3], device=device, dtype=torch.float32)
    y = torch.tensor([1.9], device=device, dtype=torch.float32)
    a_tri = torch.tensor([0], device=device, dtype=torch.float32)
    b_tri = torch.tensor([0], device=device, dtype=torch.float32)
    c_tri = torch.tensor([0], device=device, dtype=torch.float32)
    for kernel in [with_fn, without_fn]:
        kernel[(1, )](x, y, a_tri, b_tri, c_tri, num_warps=1)
        a_ref, b_ref, c_ref = x + y, x - y, x * y
        assert a_tri == a_ref
        assert b_tri == b_ref
        assert c_tri == c_ref


@triton.jit(noinline=True)
def noinline_simple_fn(x, y, Z):
    z = x + y
    tl.store(Z, z)


@triton.jit(noinline=True)
def noinline_call_graph_fn1(x):
    return x + 1


@triton.jit(noinline=True)
def noinline_call_graph_fn2(y):
    return y + 2


@triton.jit(noinline=True)
def noinline_call_graph_fn(x, y, Z):
    t0 = noinline_call_graph_fn1(x)
    t1 = noinline_call_graph_fn2(y)
    z = t0 + t1
    tl.store(Z, z)


@triton.jit(noinline=True)
def noinline_shared_fn(x, y, Z):
    offs = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    z = tl.load(Z + offs)
    z = tl.dot(z, z) + x + y
    tl.store(Z + offs, z)


@triton.jit(noinline=True)
def noinline_dynamic_fn(x, y, Z):
    if x >= 1:
        x = noinline_call_graph_fn1(x)
    else:
        x = noinline_call_graph_fn2(x)
    if y >= 2:
        y = noinline_call_graph_fn2(y)
    else:
        y = noinline_call_graph_fn1(y)
    z = x + y
    tl.store(Z, z)


@triton.jit(noinline=True)
def noinline_call_multi_values_fn(x, y):
    return x + 1, y + 2


@triton.jit(noinline=True)
def noinline_multi_values_fn(x, y, Z):
    x, y = noinline_call_multi_values_fn(x, y)
    z = x + y
    tl.store(Z, z)


@pytest.mark.interpreter
@pytest.mark.parametrize("mode", ["simple", "call_graph", "shared", "dynamic", "multi_values"])
def test_noinline(mode, device):

    @triton.jit
    def kernel(X, Y, Z):
        x = tl.load(X)
        y = tl.load(Y)
        GENERATE_TEST_HERE(x, y, Z)

    func_name = f'noinline_{mode}_fn'
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': func_name})
    x = torch.tensor([1.0], device=device, dtype=torch.float32)
    y = torch.tensor([2.0], device=device, dtype=torch.float32)
    if mode == "shared":
        z = torch.ones((16, 16), device=device, dtype=torch.float32)
    else:
        z = torch.tensor([0.0], device=device, dtype=torch.float32)
    kernel[(1, )](x, y, z, num_warps=1)
    if mode == "simple":
        assert torch.equal(z, x + y)
    elif mode == "call_graph" or mode == "dynamic" or mode == "multi_values":
        assert torch.equal(z, x + 1 + y + 2)
    elif mode == "shared":
        ref = torch.full((16, 16), 16, device=device, dtype=torch.float32)
        assert torch.equal(z, ref + x + y)


# ---------------
# test atomics
# ---------------
@pytest.mark.interpreter
@pytest.mark.parametrize(
    "op, dtype_x_str, mode, sem",
    itertools.chain.from_iterable([[
        ('add', 'bfloat16', mode, sem),
        ('add', 'float16', mode, sem),
        ('add', 'uint32', mode, sem),
        ('add', 'int32', mode, sem),
        ('add', 'float32', mode, sem),
        ('add', 'uint64', mode, sem),
        ('add', 'int64', mode, sem),
        ('add', 'float64', mode, sem),
        ('max', 'uint32', mode, sem),
        ('max', 'int32', mode, sem),
        ('max', 'float32', mode, sem),
        ('max', 'uint64', mode, sem),
        ('max', 'int64', mode, sem),
        ('max', 'float64', mode, sem),
        ('min', 'uint32', mode, sem),
        ('min', 'int32', mode, sem),
        ('min', 'float32', mode, sem),
        ('min', 'uint64', mode, sem),
        ('min', 'int64', mode, sem),
        ('min', 'float64', mode, sem),
    ]
                                   for mode in ['all_neg', 'all_pos', 'min_neg', 'max_pos']
                                   for sem in [None, 'acquire', 'release', 'acq_rel', 'relaxed']]))
def test_atomic_rmw(op, dtype_x_str, mode, sem, device):
    check_type_supported(dtype_x_str, device)
    if is_interpreter():
        if dtype_x_str == 'float16':
            pytest.skip("Only test atomic float16 ops on GPU")

    n_programs = 5

    # triton kernel
    @triton.jit
    def kernel(X, Z):
        pid = tl.program_id(0)
        x = tl.load(X + pid)
        old = GENERATE_TEST_HERE
        tl.static_assert(old.dtype == x.dtype)

    sem_arg = sem if sem is None else f'"{sem}"'
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.atomic_{op}(Z, x, sem={sem_arg})'})
    numpy_op = {'add': np.sum, 'max': np.max, 'min': np.min}[op]
    dtype_x_np = np.float32 if dtype_x_str == 'bfloat16' else getattr(np, dtype_x_str)

    max_neutral = float('-inf') if dtype_x_str in float_dtypes_with_bfloat16 else np.iinfo(dtype_x_np).min
    min_neutral = float('inf') if dtype_x_str in float_dtypes_with_bfloat16 else np.iinfo(dtype_x_np).max
    neutral = {'add': 0, 'max': max_neutral, 'min': min_neutral}[op]

    # triton result
    rs = RandomState(17)
    x = np.array([2**i for i in range(n_programs)], dtype=dtype_x_np)
    if mode == 'all_neg':
        x = -np.abs(x)
    if mode == 'all_pos':
        x = np.abs(x)
    if mode == 'min_neg':
        idx = rs.randint(n_programs, size=(1, )).item()
        x[idx] = -np.max(np.abs(x)) - 1
    if mode == 'max_pos':
        idx = rs.randint(n_programs, size=(1, )).item()
        x[idx] = np.max(np.abs(x)) + 1
    x_tri = to_triton(x, dst_type=dtype_x_str, device=device)

    z_tri = to_triton(np.array([neutral], dtype=dtype_x_np), dst_type=dtype_x_str, device=device)
    h = kernel[(n_programs, )](x_tri, z_tri)
    # torch result
    z_ref = numpy_op(x).astype(dtype_x_np)
    # compare
    exact = op not in ['add']

    # Do not check accuracy for cuda arch in the case of bf16
    if dtype_x_str != 'bfloat16' or not is_cuda():
        if exact:
            assert z_ref.item() == to_numpy(z_tri).item()
        else:
            np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)
    sem_str = "acq_rel" if sem is None else sem
    if not is_cuda():
        return

    assert f"atom.global.gpu.{sem_str}" in h.asm["ptx"]


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_atomic_rmw_predicate(num_ctas, device):

    @triton.jit
    def kernel(X):
        val = tl.program_id(0)
        if val < 64:
            tl.atomic_max(X, val)

    x = torch.zeros((1, ), device=device, dtype=torch.int32)
    kernel[(4096, )](x, num_ctas=num_ctas)
    assert x.item() == 63


@pytest.mark.interpreter
@pytest.mark.parametrize("shape, axis, num_ctas, dtype_x_str, check_return_val",
                         [(shape, axis, num_ctas, dtype_x_str, check_return_val)
                          for shape in [(2, 2), (2, 8), (8, 2), (8, 8), (32, 32), (64, 64)]
                          for axis in [0, 1]
                          for num_ctas in num_ctas_list
                          for dtype_x_str in ['bfloat16', 'float16', 'float32', 'uint64', 'int64', 'float64']
                          for check_return_val in ([True, False] if is_hip() else [True])])
def test_tensor_atomic_rmw(shape, axis, num_ctas, dtype_x_str, check_return_val, device):
    check_type_supported(dtype_x_str, device)
    shape0, shape1 = shape
    # triton kernel

    @triton.jit
    def kernel(Z, X, OLD, AXIS: tl.constexpr, SHAPE0: tl.constexpr, SHAPE1: tl.constexpr, DTYPE: tl.constexpr,
               RETURN_VAL: tl.constexpr):
        off0 = tl.arange(0, SHAPE0)
        off1 = tl.arange(0, SHAPE1)
        x = tl.load(X + off0[:, None] * SHAPE1 + off1[None, :])

        if DTYPE == tl.float16:
            # sum can have bad numerics when accumulating in float16.
            # if we're dealing with float16, do the sum in float32.
            x = x.to(tl.float32)

        z = tl.sum(x, axis=AXIS)

        if DTYPE == tl.float16:
            z = z.to(DTYPE)

        if AXIS == 1:
            old = tl.atomic_add(Z + off0, z)
            if RETURN_VAL:
                tl.store(OLD + off0, old)
        else:
            old = tl.atomic_add(Z + off1, z)
            if RETURN_VAL:
                tl.store(OLD + off1, old)

    dtype_x_np = np.float32 if dtype_x_str == 'bfloat16' else getattr(np, dtype_x_str)

    rs = RandomState(17)
    x = numpy_random((shape0, shape1), dtype_str=dtype_x_str, rs=rs)
    z_shape = (shape0, ) if axis == 1 else (shape1, )
    z = numpy_random(z_shape, dtype_str=dtype_x_str, rs=rs)
    old = np.zeros(z_shape, dtype=dtype_x_np)
    # reference results
    if x.dtype == np.float16:
        # do the sum in float32 to reduce numerical variation
        z_ref = z + np.sum(x.astype(np.float32), axis=axis, keepdims=False).astype(x.dtype)
    else:
        z_ref = z + np.sum(x, axis=axis, keepdims=False)
    old_ref = np.copy(z)
    # triton result
    x_tri = to_triton(x, device=device)
    z_tri = to_triton(z, device=device)
    old_tri = to_triton(old, device=device)

    def torch_to_triton_dtype(t):
        if t == torch.float16:
            return tl.float16
        return None

    kernel[(1, )](z_tri, x_tri, old_tri, axis, shape0, shape1, torch_to_triton_dtype(x_tri.dtype), check_return_val,
                  num_ctas=num_ctas)
    rtol = 0.01 if dtype_x_str == 'bfloat16' else 1e-4
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=rtol)
    if check_return_val:
        np.testing.assert_equal(old_ref, to_numpy(old_tri))


@pytest.mark.interpreter
@pytest.mark.parametrize("size, num_ctas, dtype_x_str", [(size, num_ctas, dtype_x_str)
                                                         for size in [2, 4, 8, 32, 64, 128]
                                                         for num_ctas in num_ctas_list
                                                         for dtype_x_str in ['bfloat16', 'float16', 'float32']])
def test_tensor_atomic_add_non_exclusive_offset(size, num_ctas, dtype_x_str, device):

    @triton.jit
    def kernel(X, val, NUM: tl.constexpr):
        off = tl.arange(0, NUM)
        offset = off[:, None] * NUM + off[None, :]
        val = tl.load(val + offset)
        tl.atomic_add(X + offset // 2, val)

    shape = (size // 2, size)
    x = torch.zeros(shape, dtype=getattr(torch, dtype_x_str), device=device)
    val = torch.randn((size**2), dtype=getattr(torch, dtype_x_str), device=device)
    kernel[(1, )](x, val, size, num_warps=1, num_ctas=num_ctas)
    ref = val[0::2] + val[1::2]

    if dtype_x_str != 'bfloat16':
        np.testing.assert_allclose(to_numpy(ref), to_numpy(x.reshape(math.prod(shape))))
    elif not is_cuda():
        np.testing.assert_allclose(to_numpy(ref), to_numpy(x.reshape(math.prod(shape))), rtol=0.01)


@pytest.mark.interpreter
@pytest.mark.parametrize("shape, idx_order, mask_step, num_ctas, dtype_x_str",
                         [(shape, idx_order, mask_step, num_ctas, dtype_x_str)
                          for shape in [(2, 2), (4, 4), (5, 5), (6, 6), (8, 8)]
                          for idx_order in ['increase', 'decrease', 'random_no_duplication', 'random']
                          for mask_step in range(1, 5)
                          for num_ctas in num_ctas_list
                          for dtype_x_str in ['bfloat16', 'float16', 'float32']])
def test_tensor_atomic_add_access_patterns(shape, idx_order, mask_step, num_ctas, dtype_x_str, device):
    check_type_supported(dtype_x_str, device)
    if is_interpreter():
        pytest.skip("not supported in the interpreter")

    @triton.jit
    def kernel(in_ptr, idx_ptr, out_ptr, shape0, shape1, mask_step, XBLOCK: tl.constexpr):
        xoffset = tl.program_id(0) * XBLOCK
        x_idx = xoffset + tl.arange(0, XBLOCK)[:]
        mask = x_idx < shape0 * shape1
        mask = mask and (x_idx % mask_step != 0)
        idx_base = shape1 * (x_idx // shape1)
        idx_offset = tl.load(idx_ptr + x_idx, mask)
        in_elem = tl.load(in_ptr + x_idx, mask)
        tl.atomic_add(out_ptr + (idx_offset + idx_base), in_elem, mask, sem='relaxed')

    shape0, shape1 = shape
    idx_row = torch.arange(0, shape1, device=device)
    if idx_order == 'increase':
        idx = torch.stack([idx_row.repeat_interleave(i + 1)[:shape1] for i in range(shape0)])
    if idx_order == 'decrease':
        idx = torch.stack([idx_row.flip(0).repeat_interleave(i + 1)[:shape1] for i in range(shape0)])
    if idx_order == 'random_no_duplication':
        idx = torch.stack([torch.randperm(shape1, device=device) for _ in idx_row])
    if idx_order == 'random':
        idx = torch.randint(0, shape1, size=(shape0, shape1), device=device)

    val = torch.randn((shape0, shape1), dtype=getattr(torch, dtype_x_str), device=device)
    dst = torch.randn((shape0, shape1), dtype=getattr(torch, dtype_x_str), device=device)

    dst_ref = dst.clone()

    cnt = 0
    for i, row in enumerate(idx):
        for j, elem in enumerate(row):
            if cnt % mask_step != 0:
                dst_ref[i][elem] += val[i][j]
            cnt += 1

    kernel[(1, )](val, idx, dst, shape0, shape1, mask_step, 64, num_ctas=num_ctas)
    if dtype_x_str != "bfloat16":
        np.testing.assert_allclose(to_numpy(dst_ref), to_numpy(dst), atol=1e-2)
    elif not is_cuda():
        np.testing.assert_allclose(to_numpy(dst_ref), to_numpy(dst), atol=0.07)


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_tensor_atomic_rmw_block(num_ctas, device):
    shape = (8, 8)

    @triton.jit
    def kernel(X, SHAPE0: tl.constexpr, SHAPE1: tl.constexpr):
        off0 = tl.arange(0, SHAPE0)
        off1 = tl.arange(0, SHAPE1)
        offs = off0[:, None] * SHAPE1 + off1[None, :]
        val = offs.to(tl.float32)
        x = X + offs
        tl.atomic_min(x, val)

    x = torch.ones((8, 8), device=device, dtype=torch.float32)
    kernel[(2, )](x, shape[0], shape[1], num_ctas=num_ctas)
    assert torch.min(x).item() == 0.0


@pytest.mark.interpreter
@pytest.mark.parametrize("sem", [None, 'acquire', 'release', 'acq_rel', 'relaxed'])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_atomic_cas(sem, num_ctas, device):
    # 1. make sure that atomic_cas changes the original value (Lock)
    @triton.jit
    def change_value(Lock):
        tl.atomic_cas(Lock, 0, 1)

    Lock = torch.zeros((1, ), device=device, dtype=torch.int32)
    change_value[(1, )](Lock)

    assert (Lock[0] == 1)

    # 2. only one block enters the critical section
    @triton.jit
    def serialized_add(data, Lock, SEM: tl.constexpr):
        ptrs = data + tl.arange(0, 128)
        while tl.atomic_cas(Lock, 0, 1, SEM) == 1:
            pass

        tl.store(ptrs, tl.load(ptrs) + 1.0)

        # insert barrier to set a fence between tl.store and
        # tl.atomic_xchg in a block.
        tl.debug_barrier()

        # release lock
        tl.atomic_xchg(Lock, 0)

    Lock = torch.zeros((1, ), device=device, dtype=torch.int32)
    data = torch.zeros((128, ), device=device, dtype=torch.float32)
    ref = torch.full((128, ), 2000.0)
    h = serialized_add[(2000, )](data, Lock, SEM=sem, num_ctas=num_ctas)
    sem_str = "acq_rel" if sem is None else sem
    np.testing.assert_allclose(to_numpy(data), to_numpy(ref))
    if not is_cuda():
        return
    assert f"atom.global.{sem_str}" in h.asm["ptx"]


@pytest.mark.interpreter
@pytest.mark.parametrize("sem", [None, 'acquire', 'release', 'acq_rel', 'relaxed'])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_tensor_atomic_cas(sem, num_ctas, device):

    @triton.jit
    def change_value(X, BLOCK_SIZE: tl.constexpr, sem: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        t1 = tl.full((BLOCK_SIZE, ), 0, dtype=tl.int64)
        t2 = tl.full((BLOCK_SIZE, ), 2, dtype=tl.int64)
        tl.atomic_cas(X + offsets, t1, t2, sem=sem)

    X = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device, dtype=torch.int64)
    Y = torch.tensor([2, 1, 2, 1, 2, 1, 2, 1], device=device, dtype=torch.int64)

    change_value[(2, )](X, 4, sem)
    assert (torch.equal(X, Y))


@pytest.mark.interpreter
@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
                    reason="Requires compute capability >= 9 for NV")
def test_load_scope_sem_coop_grid_cta_not_one(device):

    @triton.jit
    def kernel_r(ptrs, BLOCK_SIZE: tl.constexpr):
        numel = 512
        offset = tl.program_id(0) * BLOCK_SIZE
        index = offset
        mask = index < numel
        a = tl.load(ptrs, mask=mask)
        tl.store(ptrs, a)

    block_size = 128
    data = torch.zeros((128, ), device=device, dtype=torch.float32)

    out = kernel_r[(2, )](data, BLOCK_SIZE=block_size, num_ctas=4, launch_cooperative_grid=True)
    out = kernel_r[(2, )](data, BLOCK_SIZE=block_size, num_ctas=4, launch_cooperative_grid=False)


@pytest.mark.interpreter
def test_load_scope_sem_coop_grid_cta_one(device):

    @triton.jit
    def kernel_r(ptrs, BLOCK_SIZE: tl.constexpr):
        numel = 512
        offset = tl.program_id(0) * BLOCK_SIZE
        index = offset
        mask = index < numel
        a = tl.load(ptrs, mask=mask)
        tl.store(ptrs, a)

    block_size = 128
    data = torch.zeros((128, ), device=device, dtype=torch.float32)

    # Should do nothing different for num_ctas=1 (with coop launch grid)
    out = kernel_r[(2, )](data, BLOCK_SIZE=block_size, num_ctas=1, launch_cooperative_grid=True)
    out = kernel_r[(2, )](data, BLOCK_SIZE=block_size, num_ctas=1, launch_cooperative_grid=False)


# ---------------
# test cast
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, dtype_z, bitcast, size",
                         [(dtype_x, dtype_z, False, 1024) for dtype_x in dtypes for dtype_z in dtypes] + [
                             ('float32', 'bfloat16', False, 1024),
                             ('bfloat16', 'float32', False, 1024),
                             ('float32', 'int32', True, 1024),
                             ('float32', 'bool', False, 1024),
                             ('int8', 'bfloat16', False, 1024),
                         ] + [(f'uint{x}', f'int{x}', True, 1024)
                              for x in [8, 16, 32, 64]] + [(f'int{x}', f'uint{x}', True, 1024)
                                                           for x in [8, 16, 32, 64]] +
                         (([(dtype_x, dtype_z, False, size)
                            for dtype_x in torch_float8_dtypes
                            for dtype_z in ["float16", "float32", "bfloat16"]
                            for size in [1024, 32]]  #
                           + [(dtype_x, dtype_z, False, size)
                              for dtype_z in torch_float8_dtypes
                              for dtype_x in ["float16", "float32", "bfloat16"]
                              for size in [1024, 32]]) if torch.__version__ >= "2.1" else []))
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_cast(dtype_x, dtype_z, bitcast, size, num_ctas, device):
    # CUDA: bfloat16 on cc < 80 will not be tested
    # Interpreter: Only bfloat16 <-> float32 is supported
    if not is_interpreter() or \
        (is_interpreter() and not ((dtype_z == 'bfloat16' and dtype_x == 'float32')
                                   or (dtype_z == 'float32' and dtype_x == 'bfloat16'))):
        check_type_supported(dtype_x, device)
        check_type_supported(dtype_z, device)

    if is_hip() and (dtype_z in ("bfloat16", "float8_e4m3fn") or dtype_x == "float8_e4m3fn"):
        pytest.skip(f'test_cast{(dtype_x, dtype_z)} cast to bfloat16 not supported on HIP.')

    torch.manual_seed(0)
    # This is tricky because numpy doesn't have bfloat, and torch doesn't have uints.
    if dtype_x.startswith('bfloat'):
        x_tri = torch.randn(size, dtype=getattr(torch, dtype_x), device=device)
    elif dtype_x.startswith('float8'):
        x_tri = torch.randn(size, dtype=torch.half, device=device).to(dtype=getattr(torch, dtype_x))
    else:
        x = numpy_random(size, dtype_str=dtype_x, low=-10, high=10) * 10
        # Triton clamps negative values to zero, while numpy wraps around
        # intmax, so avoid negatives for now.
        # TODO: figure out which one should actually be happening, and test it
        if dtype_z in uint_dtypes:
            x = np.absolute(x)
        x_tri = to_triton(x, device=device)
    if 'float' in dtype_z and 'float' in dtype_x:
        # make sure we use values that can be represented in both types
        x_tri = x_tri.to(getattr(torch, dtype_z)).to(getattr(torch, dtype_x))
    # triton kernel

    @triton.jit
    def kernel(X, Z, TO_TYPE: tl.constexpr, BITCAST: tl.constexpr, SIZE: tl.constexpr, ARG_HASH: tl.constexpr):
        x_ptr = X + tl.arange(0, SIZE)
        z_ptr = Z + tl.arange(0, SIZE)
        x = tl.load(x_ptr)

        # Depending on the value of ARG_HASH (a "random" number determined by
        # the test parameters), spell the cast one of three different ways.
        if ARG_HASH % 4 == 0:
            z = x.to(Z.dtype.element_ty, bitcast=BITCAST)
        elif ARG_HASH % 4 == 1:
            z = x.cast(Z.dtype.element_ty, bitcast=BITCAST)
        elif ARG_HASH % 4 == 2:
            z = tl.cast(x, Z.dtype.element_ty, bitcast=BITCAST)
        else:
            z = tl.cast(x, TO_TYPE, bitcast=BITCAST)

        tl.store(z_ptr, z)

    # "Random" number used inside the kernel to determine how we spell the cast.
    # This way we don't have to increase the number of tests.
    arg_hash = hash((dtype_x, dtype_z, bitcast, size, num_ctas))

    dtype_z_np = dtype_z if dtype_z != 'bool' else 'bool_'
    # triton result
    if dtype_z.startswith('bfloat'):
        z_tri = torch.empty((size, ), dtype=getattr(torch, dtype_z), device=device)
    elif dtype_z.startswith('float8'):
        z_tri = torch.empty((size, ), dtype=torch.half, device=device).to(dtype=getattr(torch, dtype_z))
    else:
        z_tri = to_triton(np.empty((size, ), dtype=getattr(np, dtype_z_np)), device=device)

    dtype_z_tri = str_to_triton_dtype(dtype_z)
    kernel[(1, )](x_tri, z_tri, TO_TYPE=dtype_z_tri, BITCAST=bitcast, SIZE=size, ARG_HASH=arg_hash, num_warps=1,
                  num_ctas=num_ctas)
    # torch result
    if dtype_z.startswith('bfloat') or dtype_x.startswith('bfloat') or dtype_z.startswith(
            'float8') or dtype_x.startswith('float8'):
        assert bitcast is False
        z_ref = x_tri.to(z_tri.dtype)
        if dtype_z.startswith('float8') and device not in ['cuda']:
            t = z_ref.byte() ^ z_tri.byte()
            torch.testing.assert_close(torch.zeros_like(t, dtype=torch.uint8), t)
        else:
            torch.testing.assert_close(z_ref, z_tri, rtol=0, atol=0)
    else:
        if bitcast:
            z_ref = x.view(getattr(np, dtype_z_np))
        else:
            z_ref = x.astype(getattr(np, dtype_z_np))
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0, atol=0)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str, num_warps",
                         [(dtype_str, num_warps) for dtype_str in int_dtypes + float_dtypes for num_warps in [4, 8]])
def test_cat(dtype_str, num_warps, device):
    check_type_supported(dtype_str, device)

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.cat(x, y, can_reorder=True)
        tl.store(Z + tl.arange(0, 2 * N), z)

    x = torch.arange(0, 128, device=device).to(getattr(torch, dtype_str))
    y = torch.arange(-128, 0, device=device).to(getattr(torch, dtype_str))
    z_ref = torch.cat([x, y], dim=0).sum()
    z = torch.zeros((256, ), dtype=getattr(torch, dtype_str), device=device)
    kernel[(1, )](x, y, z, N=128, num_warps=num_warps)
    assert z.sum() == z_ref
    # check if there's no duplicate value in z
    assert z.unique().size(0) == z.size(0)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", list(torch_dtypes))
@pytest.mark.parametrize("constant_field", ["value", "mask"])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_store_constant(num_ctas, dtype_str, constant_field, device):
    check_type_supported(dtype_str, device)

    @triton.jit
    def kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, CONSTANT_FIELD: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if CONSTANT_FIELD == "value":
            value = 1
            output = tl.full([BLOCK_SIZE], value=value, dtype=value.dtype)
            mask = offsets < n_elements
        elif CONSTANT_FIELD == "mask":
            output = offsets < n_elements
            mask = False
        tl.store(output_ptr + offsets, output, mask=mask)

    block_size = 128
    ref = torch.ones([block_size], dtype=getattr(torch, dtype_str), device=device)
    output = torch.zeros([block_size], dtype=getattr(torch, dtype_str), device=device)

    kernel[(1, )](output, block_size, BLOCK_SIZE=block_size, num_ctas=num_ctas, CONSTANT_FIELD=constant_field)

    if constant_field == "value":
        assert torch.all(output == ref)
    else:
        assert torch.all(output == 0)


def test_load_store_same_ptr(device):

    @triton.jit()
    def kernel(in_out_ptr):
        pid = tl.program_id(axis=0)
        x = tl.load(in_out_ptr + pid)
        out = x * 2
        tl.store(in_out_ptr + pid, out)

    for _ in range(1000):
        x = torch.ones((65536, ), device=device, dtype=torch.float32)
        if is_hip():
            kernel[(65536, )](x, num_warps=16)  # threads per Warp for ROCM is 64
        else:
            kernel[(65536, )](x, num_warps=32)
        assert torch.all(x == 2)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ['int32'])
def test_umulhi(dtype_str, device):

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.umulhi(x, y)
        tl.store(Z + tl.arange(0, N), z)

    def umulhi32(a, b):
        # Convert to 64-bit unsigned integers to prevent overflow
        a_64 = a.astype(np.int64)
        b_64 = b.astype(np.int64)

        # Perform the multiplication in 64-bit
        product_64 = a_64 * b_64

        # Shift right by 32 bits to get the high part of the product
        result_high_32 = product_64 >> 32
        return result_high_32

    rs = RandomState(17)
    N = 128
    x = numpy_random((N, ), dtype_str=dtype_str, rs=rs, low=0)
    x_tri = to_triton(x, device=device)
    y = numpy_random((N, ), dtype_str=dtype_str, rs=rs, low=0)
    y_tri = to_triton(y, device=device)
    z_tri = torch.zeros_like(x_tri)
    kernel[(1, )](x_tri, y_tri, z_tri, N=N)

    z_ref = umulhi32(x, y)
    np.testing.assert_equal(z_ref, to_numpy(z_tri))


@pytest.mark.interpreter
def test_join(device):

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.join(x, y)
        tl.store(Z + tl.arange(0, N)[:, None] * 2 + tl.arange(0, 2)[None, :], z)

    x = torch.arange(0, 128, device=device).to(torch.int32)
    y = torch.arange(-128, 0, device=device).to(torch.int32)
    z_ref = torch.stack([x, y], dim=-1)
    z = torch.zeros_like(z_ref)
    kernel[(1, )](x, y, z, N=128)

    np.testing.assert_equal(to_numpy(z_ref), to_numpy(z))


@pytest.mark.interpreter
def test_join_scalars(device):

    @triton.jit
    def kernel(X, Y, Z):
        x = tl.load(X)
        y = tl.load(Y)
        z = tl.join(x, y)
        tl.static_assert(z.shape == [2])
        tl.store(Z + tl.arange(0, 2), z)

    x = torch.full([1], 42, device=device).to(torch.int32)
    y = torch.full([1], 100, device=device).to(torch.int32)
    z = torch.zeros([2], device=device)
    kernel[(1, )](x, y, z)

    np.testing.assert_equal([42, 100], to_numpy(z))


@pytest.mark.interpreter
def test_join_with_mma(device):

    @triton.jit
    def kernel(X, Z):
        x = tl.load(X + 16 * tl.arange(0, 32)[:, None] + tl.arange(0, 16)[None, :])  # (32,16)
        x2 = tl.join(x, 2 * x)  # (32,16,2)
        x3 = tl.reshape(x2, (32, 32))
        z = tl.dot(x3, x3)  # (32,32)
        tl.store(Z + 32 * tl.arange(0, 32)[:, None] + tl.arange(0, 32)[None, :], z)

    x = torch.arange(0, 32 * 16, device=device, dtype=torch.float32).reshape((32, 16))
    r = torch.stack([x, 2 * x], dim=-1).reshape((32, 32))
    z_ref = torch.matmul(r, r)
    z = torch.zeros_like(z_ref)
    kernel[(1, )](x, z)

    torch.testing.assert_close(z, z_ref)


@pytest.mark.interpreter
@pytest.mark.parametrize("debug", [False, True])
def test_interleave(device, debug):

    @triton.jit(debug=debug)
    def kernel(Z, N: tl.constexpr):
        z = tl.interleave(tl.arange(0, N), tl.arange(N, 2 * N))
        tl.store(Z + tl.arange(0, 2 * N), z)

    x = torch.arange(0, 128, device=device).to(torch.int32)
    y = torch.arange(128, 256, device=device).to(torch.int32)
    z_ref = torch.stack([x, y], dim=-1).reshape(256)
    z = torch.zeros_like(z_ref)
    kernel[(1, )](z, N=128)

    np.testing.assert_equal(to_numpy(z_ref), to_numpy(z))


@pytest.mark.interpreter
def test_interleave_scalars(device):

    @triton.jit
    def kernel(X, Y, Z):
        z = tl.interleave(X, Y)
        tl.static_assert(z.shape == [tl.constexpr(2)])
        tl.store(Z + tl.arange(0, 2), z)

    z = torch.zeros(2, device=device)
    kernel[(1, )](10, 20, z)

    np.testing.assert_equal([10, 20], to_numpy(z))


@pytest.mark.interpreter
def test_split(device):

    @triton.jit
    def kernel(X, Z1, Z2, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        x1 = tl.reshape(x, (N // 2, 2))
        z1, z2 = tl.split(x1)
        tl.store(Z1 + tl.arange(0, N // 2), z1)
        tl.store(Z2 + tl.arange(0, N // 2), z2)

    x = torch.arange(0, 256, device=device).to(torch.int32).reshape((128, 2))
    z1_ref, z2_ref = (x[:, 0], x[:, 1])
    z1 = torch.zeros_like(z1_ref)
    z2 = torch.zeros_like(z2_ref)
    kernel[(1, )](x, z1, z2, N=256)

    np.testing.assert_equal(to_numpy(z1_ref), to_numpy(z1))
    np.testing.assert_equal(to_numpy(z2_ref), to_numpy(z2))


@pytest.mark.interpreter
def test_split_to_scalar(device):

    @triton.jit
    def kernel(X, Z1, Z2):
        offs = tl.arange(0, 2)
        x = tl.load(X + offs)
        z1, z2 = tl.split(x)
        tl.static_assert(isinstance(z1, tl.tensor))
        tl.static_assert(isinstance(z2, tl.tensor))
        tl.static_assert(z1.shape == [])
        tl.static_assert(z2.shape == [])
        tl.store(Z1, z1)
        tl.store(Z2, z2)

    N = 2
    x = torch.arange(0, N, device=device).reshape(N // 2, 2)
    z1_ref, z2_ref = (x[:, 0], x[:, 1])
    z1 = torch.zeros_like(z1_ref)
    z2 = torch.zeros_like(z2_ref)
    kernel[(1, )](x, z1, z2)

    np.testing.assert_equal(to_numpy(z1_ref), to_numpy(z1))
    np.testing.assert_equal(to_numpy(z2_ref), to_numpy(z2))


def convert_float_to_float32(fp: torch.tensor, dtype=None):
    if not dtype:
        dtype = getattr(tl, torch_dtype_name(fp.dtype))

    fp = fp.view(getattr(torch, f"int{dtype.primitive_bitwidth}"))
    exp_width = dtype.primitive_bitwidth - dtype.fp_mantissa_width - 1
    exp_bias = dtype.exponent_bias
    sign = ((fp >> (dtype.primitive_bitwidth - 1)) & 0x01).int()
    exp = ((fp >> dtype.fp_mantissa_width) & ((1 << exp_width) - 1)).int()
    frac = (fp & ((1 << dtype.fp_mantissa_width) - 1)).int()

    output = torch.where(
        exp == 0,
        # subnormal
        ((-1.0)**sign) * (2.0**(1 - exp_bias)) * (frac / (2.0**dtype.fp_mantissa_width)),
        # normal
        ((-1.0)**sign) * (2.0**(exp - exp_bias)) * (1.0 + frac / (2.0**dtype.fp_mantissa_width))).float()

    extended_exp = (
        (1 << (tl.float32.primitive_bitwidth - tl.float32.fp_mantissa_width - 1)) - 1) << tl.float32.fp_mantissa_width
    # special cases, exp is 0b11..1
    if dtype in [tl.float8e4nv, tl.float8e4b15]:
        # float8e4m3nv does not have infinities
        output[fp == 0b01111111] = torch.nan
        output[fp == 0b11111111] = torch.nan
    else:
        output = torch.where(exp == (1 << exp_width) - 1,
                             ((sign << (tl.float32.primitive_bitwidth - 1)) | extended_exp
                              | (frac << (tl.float32.fp_mantissa_width - dtype.fp_mantissa_width)))  #
                             .view(torch.float32), output)
    return output


@pytest.mark.interpreter
@pytest.mark.parametrize("in_dtype", [torch.float16, torch.bfloat16])
def test_convert_float16_to_float32(in_dtype, device):
    """Tests that check convert_float_to_float32 function"""
    check_type_supported(in_dtype, device)

    f16_input = torch.tensor(range(-int(2**(16 - 1)), int(2**(16 - 1))), dtype=torch.int16).view(in_dtype)
    f32_output = convert_float_to_float32(f16_input)

    nan = f16_input.isnan()
    assert torch.all(f32_output[nan].isnan())
    inf = f16_input.isinf()
    assert torch.all(f32_output[inf].isinf())
    other = torch.logical_not(torch.logical_or(nan, inf))
    assert torch.all(f16_input[other] == f32_output[other])


def serialize_fp8(np_data, in_dtype):
    return np_data


# inverse of `serialize_fp8`


def deserialize_fp8(np_data, in_dtype):
    return np_data


# ---------------
# test reduce
# ---------------


@pytest.mark.interpreter
def test_max_returns_zero(device):
    # Simple test with a tl.max call that returns 0.  The interpreter had a bug
    # where it didn't handle this correctly.
    @triton.jit
    def kernel(X, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        z = tl.max(x)
        tl.store(Z, z)

    BLOCK = 128
    x = torch.zeros((BLOCK, ), device=device)
    z = torch.ones((1, ), device=device)

    kernel[(1, )](x, z, BLOCK=BLOCK)
    assert z[0] == 0


def get_reduced_dtype(dtype_str, op):
    if op in ('argmin', 'argmax'):
        return 'int32'
    if dtype_str == 'bfloat16':
        return 'float32'
    return dtype_str


@pytest.mark.interpreter
@pytest.mark.parametrize("op, dtype_str, shape", [(op, dtype, shape) for op in [
    'min',
    'max',
    'min-with-indices',
    'max-with-indices',
    'argmin-tie-break-left',
    'argmax-tie-break-left',
    'sum',
] for dtype in dtypes_with_bfloat16 for shape in [32, 64, 128, 512]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_reduce1d(op, dtype_str, shape, num_ctas, device):
    check_type_supported(dtype_str, device)  # bfloat16 on cc < 80 will not be tested

    # triton kernel
    @triton.jit
    def kernel(X, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        GENERATE_TEST_HERE
        tl.store(Z, z)

    if 'with-indices' in op:
        patch = f'z, _ = tl.{op.split("-")[0]}(x, axis=0, return_indices=True)'
    elif 'arg' in op:
        tie_break_left = 'tie-break-left' in op
        patch = f'z = tl.{op.split("-")[0]}(x, axis=0, tie_break_left={tie_break_left})'
    else:
        patch = f'z = tl.{op}(x, axis=0)'
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': patch})
    # input
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random((shape, ), dtype_str=dtype_str, rs=rs)
    numpy_op = {
        'sum': np.sum,
        'max': np.max,
        'min': np.min,
        'max-with-indices': np.max,
        'min-with-indices': np.min,
        'argmin-tie-break-left': np.argmin,
        'argmax-tie-break-left': np.argmax,
    }[op]
    if 'tie-break-left' in op:
        x[3:10] = x[numpy_op(x)]
    x_tri = to_triton(x, device=device)
    # numpy result
    z_dtype_str = 'int32' if 'tie-break-left' in op else dtype_str
    z_tri_dtype_str = z_dtype_str
    if 'tie-break-left' not in op and dtype_str == 'bfloat16':
        z_dtype_str = 'float32'
        z_ref = numpy_op(x).astype(getattr(np, z_dtype_str))
        # trunc mantissa for a fair comparison of accuracy
        z_ref = (z_ref.view('uint32') & np.uint32(0xffff0000)).view('float32')
        z_tri_dtype_str = 'bfloat16'
    else:
        z_ref = numpy_op(x).astype(getattr(np, z_dtype_str))
    # triton result
    z_tri = to_triton(numpy_random((1, ), dtype_str=z_dtype_str, rs=rs), device=device, dst_type=z_tri_dtype_str)
    kernel[(1, )](x_tri, z_tri, BLOCK=shape, num_ctas=num_ctas)
    z_tri = to_numpy(z_tri)
    # compare
    if op == 'sum':
        np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        if 'tie-break-left' in op:
            # argmin and argmax can have multiple valid indices.
            # so instead we compare the values pointed by indices
            np.testing.assert_equal(x[z_ref], x[z_tri])
        else:
            np.testing.assert_equal(z_ref, z_tri)


# TODO: [Qingyi] Fix argmin / argmax
reduce_configs1 = [(op, dtype, (1, 1024), axis, False)
                   for dtype in dtypes_with_bfloat16
                   for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                   for axis in [1]]

# shape (128, 256) and (32, 1024) are not enabled on sm86 because the required shared memory
# exceeds the limit of 99KB
reduce2d_shapes = [(2, 32), (4, 32), (4, 128)]
# TODO: fix and uncomment
# , (32, 64), (64, 128)]
if is_cuda() and 'V100' in torch.cuda.get_device_name(0):
    reduce2d_shapes += [(128, 256) and (32, 1024)]

reduce_configs2 = [(op, 'float32', shape, axis, False)
                   for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                   for shape in reduce2d_shapes
                   for axis in [0, 1]] + [(op, 'float32', [16, 32], None, False) for op in ['min', 'max', 'sum']]

reduce3d_shapes = [(2, 32, 16), (32, 2, 16), (32, 16, 2)]
reduce_configs3 = [(op, 'float32', shape, axis, False)
                   for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                   for shape in reduce3d_shapes
                   for axis in [0, 1, 2]]
invalid_config = [('sum', 'float32', (32, 32), axis, False) for axis in [2, 3]]
negative_config = [('sum', 'float32', (32, 32), -1, False)]
keep_dims_2d_configs = [(op, 'float32', (32, 32), axis, True)
                        for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                        for axis in [0, 1]] + [(op, 'float32', (32, 32), None, True) for op in ['min', 'max', 'sum']]
keep_dims_3d_configs = [(op, 'float32', (32, 2, 16), axis, True)
                        for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                        for axis in [0, 1, 2]] + [(op, 'float32', (32, 2, 16), None, True)
                                                  for op in ['min', 'max', 'sum']]
reduce_bool = [(op, 'bool', shape, axis, False) for op in ['xor_sum'] for shape in reduce2d_shapes for axis in [0, 1]]


@pytest.mark.interpreter
@pytest.mark.parametrize(
    "op, dtype_str, shape, axis, keep_dims", reduce_configs1 + reduce_configs2 + reduce_configs3 + invalid_config +
    negative_config + keep_dims_2d_configs + keep_dims_3d_configs + reduce_bool)
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_reduce(op, dtype_str, shape, axis, keep_dims, num_ctas, device):
    check_type_supported(dtype_str, device)  # bfloat16 on cc < 80 will not be tested

    @triton.jit
    def kernel(X, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, IS_3D: tl.constexpr,
               AXIS: tl.constexpr, KEEP_DIMS: tl.constexpr, USE_I1: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        range_k = tl.arange(0, BLOCK_K)
        if IS_3D:
            x = tl.load(X + range_m[:, None, None] * BLOCK_N * BLOCK_K + range_n[None, :, None] * BLOCK_K +
                        range_k[None, None, :])
        else:
            x = tl.load(X + range_m[:, None] * BLOCK_N + range_n[None, :])
        if USE_I1:
            x = tl.cast(x, tl.int1)
        z = GENERATE_TEST_HERE
        z_ptr = Z
        if KEEP_DIMS and AXIS is None:
            if IS_3D:
                z_ptr = z_ptr[None, None, None, :]
            else:
                z_ptr = z_ptr[None, None, :]
        if IS_3D:
            if AXIS == 0:
                z_ptr = Z + range_n[:, None] * BLOCK_K + range_k[None, :]
            elif AXIS == 1 or AXIS == -2:
                z_ptr = Z + range_m[:, None] * BLOCK_K + range_k[None, :]
            elif AXIS == 2 or AXIS == -1:
                z_ptr = Z + range_m[:, None] * BLOCK_N + range_n[None, :]
        else:
            if AXIS == 0:
                z_ptr = Z + range_n
            elif AXIS == 1 or AXIS == -1:
                z_ptr = Z + range_m
        if KEEP_DIMS and AXIS is not None:
            z_ptr = tl.expand_dims(z_ptr, axis=AXIS)
        tl.store(z_ptr, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{op}(x, axis=AXIS, keep_dims=KEEP_DIMS)'})
    # input
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random(shape, dtype_str=dtype_str, rs=rs)
    x_tri = to_triton(x, device=device)
    numpy_op = {
        'sum': np.sum, 'max': np.max, 'min': np.min, 'argmin': np.argmin, 'argmax': np.argmax, 'xor_sum':
        np.bitwise_xor.reduce
    }[op]
    z_dtype_str = get_reduced_dtype(dtype_str, op)
    z_tri_dtype_str = z_dtype_str
    if z_dtype_str == 'bool':
        z_dtype_str = 'int8'

    # numpy result
    # Silence numpy error on axis out of bounds, to give triton a chance to fail
    np_axis = axis if axis is not None and axis < len(shape) else None
    if op not in ['argmin', 'argmax'] and dtype_str == 'bfloat16':
        z_dtype_str = 'float32'
        z_tri_dtype_str = 'bfloat16'
        z_ref = numpy_op(x, axis=np_axis, keepdims=keep_dims).astype(getattr(np, z_dtype_str))
        # trunc mantissa for a fair comparison of accuracy
        z_ref = (z_ref.view('uint32') & np.uint32(0xffff0000)).view('float32')
    else:
        z_ref = numpy_op(x, axis=np_axis, keepdims=keep_dims).astype(getattr(np, z_dtype_str))

    # triton result
    z_shape = z_ref.shape
    z_tri = to_triton(numpy_random(z_shape, dtype_str=z_dtype_str, rs=rs), device=device, dst_type=z_tri_dtype_str)
    BLOCK_K = 1 if len(shape) == 2 else shape[2]
    IS_3D = bool(len(shape) == 3)
    USE_I1 = dtype_str == 'bool'
    if axis is not None and axis >= len(shape):
        with pytest.raises(triton.TritonError):
            kernel[(1, )](x_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], BLOCK_K=BLOCK_K, IS_3D=IS_3D, AXIS=axis,
                          KEEP_DIMS=keep_dims, USE_I1=USE_I1, num_ctas=num_ctas)
        return
    else:
        kernel[(1, )](x_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], BLOCK_K=BLOCK_K, IS_3D=IS_3D, AXIS=axis,
                      KEEP_DIMS=keep_dims, USE_I1=USE_I1, num_ctas=num_ctas)

    z_tri = to_numpy(z_tri)

    # compare
    if op == 'sum':
        np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        if op in ('argmin', 'argmax'):
            # argmin and argmax can have multiple valid indices.
            # so instead we compare the values pointed by indices
            z_ref_index = z_ref
            z_tri_index = z_tri
            if not keep_dims:
                z_ref_index = np.expand_dims(z_ref, axis=axis)
                z_tri_index = np.expand_dims(z_tri, axis=axis)
            z_ref_value = np.take_along_axis(x, z_ref_index, axis=axis)
            z_tri_value = np.take_along_axis(x, z_tri_index, axis=axis)
            np.testing.assert_equal(z_ref_value, z_tri_value)
        else:
            np.testing.assert_equal(z_ref, z_tri)


scan2d_shapes = [(8, 32), (16, 32), (32, 16), (2, 1024), (1024, 2), (32, 32), (1, 1024)]

scan_configs = [(op, type, shape, axis, reverse, num_warps)
                for num_warps in [4, 16]
                for type in ['int32', 'float32', 'bfloat16']
                for axis in [1, 0]
                for reverse in [True, False]
                for shape in scan2d_shapes
                for op in ['cumsum', 'cumprod', 'get_first_element', 'linear_recurrence', 'cummax', 'roll']]
negative_config = [('cumsum', 'float32', (32, 32), -1, False, 4)]


def test_sum_dtype(device):

    @triton.jit
    def kernel_dtype(out_ptr, init, in_dtype: tl.constexpr, out_dtype: tl.constexpr):
        x = tl.full((32, 32), init, dtype=in_dtype)
        x = tl.sum(x, dtype=out_dtype)
        tl.store(out_ptr, x.to(tl.int32))

    @triton.jit
    def kernel_default_int(out_ptr):
        x = tl.full((32, 32), 1, dtype=tl.int1)
        x = tl.sum(x)
        tl.store(out_ptr, x)

    @triton.jit
    def kernel_default_float(out_ptr):
        x = tl.full((32, 32), 1.0, dtype=tl.bfloat16)
        x = tl.sum(x)
        tl.store(out_ptr, x)

    out = torch.empty(1, dtype=torch.int32, device=device)
    kernel_dtype[(1, )](out, init=1, in_dtype=tl.int1, out_dtype=None)
    assert out[0] == 32 * 32

    kernel_dtype[(1, )](out, init=1, in_dtype=tl.int1, out_dtype=tl.int1)
    assert out[0] == 0

    kernel_dtype[(1, )](out, init=7, in_dtype=tl.int8, out_dtype=tl.int8)
    assert out[0] == (7 * 32 * 32) % 256

    kernel_dtype[(1, )](out, init=1, in_dtype=tl.int32, out_dtype=None)
    assert out[0] == 32 * 32

    kernel_default_int[(1, )](out)
    assert out[0] == 32 * 32

    out = torch.empty(1, dtype=torch.bfloat16, device=device)
    kernel_default_float[(1, )](out)
    torch.testing.assert_close(out[0], torch.tensor(32 * 32, dtype=torch.bfloat16, device=device))


@triton.jit
# trivial associative but not commutative function
def get_first_element(a, b):
    return a


# Compute x_i = a_i * x_{i-1} + b_i
@triton.jit
def linear_recurrence(a1, b1, a2, b2):
    return a1 * a2, b1 * a2 + b2


@triton.jit
def cummax(v0, i0, v1, i1):
    gt = v0 > v1
    return tl.where(gt, v0, v1), tl.where(gt, i0, i1)


@triton.jit
def roll(a1, b1_last, b1_cur, a2, b2_last, b2_cur):
    return a1 + a2, tl.where(a2 == 1, b1_cur, 0) + b2_last, b2_cur


@pytest.mark.interpreter
@pytest.mark.parametrize("op, dtype_str, shape, axis, reverse, num_warps", scan_configs + negative_config)
def test_scan2d(op, dtype_str, shape, axis, reverse, num_warps, device):
    check_type_supported(dtype_str, device)
    if dtype_str == 'bfloat16':
        if op == 'cummax':
            pytest.skip("bfloat16 compare not supported before sm90")
        if op == 'linear_recurrence':
            pytest.skip("Skipping linear_recurrence scan on bfloat16 due to accuracy issues")
    numpy_dtype_str = 'float32' if dtype_str == 'bfloat16' else dtype_str

    # triton kernel
    @triton.jit
    def kernel(X, Y, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        x = tl.load(X + range_m[:, None] * BLOCK_N + range_n[None, :])
        y = tl.load(Y + range_m[:, None] * BLOCK_N + range_n[None, :])
        GENERATE_TEST_HERE
        tl.store(Z + range_m[:, None] * BLOCK_N + range_n[None, :], z)

    if op == 'cumsum' or op == 'cumprod':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'z = tl.{op}(x, axis={axis}, reverse={reverse})'})
    elif op == 'get_first_element':
        kernel = patch_kernel(
            kernel,
            {'GENERATE_TEST_HERE': f'z = tl.associative_scan(x, axis={axis}, combine_fn={op}, reverse={reverse})'})
    elif op == 'cummax':
        rg = "range_m[:, None]" if axis == 0 else "range_n[None, :]"
        rg = f"tl.broadcast_to({rg}.to(tl.int64), [BLOCK_M, BLOCK_N])"
        kernel = patch_kernel(kernel, {
            'GENERATE_TEST_HERE':
            f'_, z = tl.associative_scan((x, {rg}), axis={axis}, combine_fn={op}, reverse={reverse})'
        })
    elif op == 'roll':
        assert op == 'roll'
        kernel = patch_kernel(
            kernel, {
                'GENERATE_TEST_HERE':
                f'_, z, _ = tl.associative_scan((1 + 0* x, 0 * x, x), axis={axis}, combine_fn={op}, reverse={reverse})'
            })
    else:
        assert op == 'linear_recurrence'
        kernel = patch_kernel(kernel, {
            'GENERATE_TEST_HERE':
            f'_, z = tl.associative_scan((x, y), axis={axis}, combine_fn={op}, reverse={reverse})'
        })
    # input
    rs = RandomState(17)
    if op == 'linear_recurrence' and dtype_str in int_dtypes:
        # If the numbers are too large the op will overflow
        # We sample numbers in -1, 0, 1
        x = rs.randint(-1, 2, shape, dtype=dtype_str)
        y = rs.randint(-1, 2, shape, dtype=dtype_str)
    else:
        x = numpy_random(shape, dtype_str=dtype_str, rs=rs)
        # y is just used in linear_recurrence
        y = numpy_random(shape, dtype_str=dtype_str, rs=rs)
    x_in = x
    if reverse:
        x_in = np.flip(x, axis)
    z = np.empty_like(x)
    x_tri = to_triton(x, device=device, dst_type=dtype_str)
    y_tri = to_triton(y, device=device, dst_type=dtype_str)
    if op == 'cumsum' or op == 'cumprod':
        numpy_op = {'cumsum': np.cumsum, 'cumprod': np.cumprod}[op]
        z_ref = numpy_op(x_in, axis=axis).astype(getattr(np, numpy_dtype_str))
        if reverse:
            z_ref = np.flip(z_ref, axis)

    elif op == 'cummax':
        # NumPy does not have cummax
        z = z.astype(np.int64)
        z_ref = torch.cummax(torch.from_numpy(x_in.copy()), axis=axis).indices.numpy()
        if reverse:
            z_ref = x_in.shape[axis] - np.flip(z_ref, axis) - 1
    elif op == 'roll':
        ROLL = 1
        z_ref = np.roll(x_in.copy(), ROLL, axis=axis)
        if axis == 0:
            z_ref[:ROLL] = 0
        else:
            z_ref[:, :ROLL] = 0

        if reverse:
            z_ref = np.flip(z_ref, axis)
    elif op == 'linear_recurrence':
        # Simplify to the axis=1 case
        x_ref = x.T if axis == 0 else x
        y_ref = y.T if axis == 0 else y
        if reverse:
            x_ref = np.flip(x_ref, 1)
            y_ref = np.flip(y_ref, 1)

        result = []
        for x_refi, y_refi in zip(x_ref, y_ref):
            li = []
            acc = 0
            for xi, yi in zip(x_refi, y_refi):
                acc = xi * acc + yi
                li.append(acc)
            result.append(li)
        z_ref = np.array(result)
        if reverse:
            z_ref = np.flip(z_ref, 1)

        if axis == 0:
            z_ref = z_ref.T
    else:
        assert op == 'get_first_element'
        z_ref = x
        if axis == 0:
            if reverse:
                z_ref[:-1] = x[-1]
            else:
                z_ref[1:] = x[0]
        else:
            if reverse:
                z_ref[:, :-1] = x[:, -1:]
            else:
                z_ref[:, 1:] = x[:, 0:1]

    # triton result
    # we don't cast the `fp32 = bf16 op bf16` result to bfloat16 to alleviate accuracy issues
    z_tri = to_triton(z, device=device)
    kernel[(1, )](x_tri, y_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], AXIS=axis, num_warps=num_warps)

    z_tri = to_numpy(z_tri)
    # compare
    if dtype_str not in int_dtypes:
        if op == 'cumprod':
            np.testing.assert_allclose(z_ref, z_tri, rtol=0.01, atol=1e-3)
        else:
            np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        np.testing.assert_equal(z_ref, z_tri)


# ---------------
# test histogram
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("M, N", [[2048, 2], [1024, 8], [1024, 128], [256, 512], [32, 512], [8, 512], [8, 2]])
def test_histogram(M, N, device):

    @triton.jit
    def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
        offset1 = tl.arange(0, M)
        offset2 = tl.arange(0, N)
        x = tl.load(x_ptr + offset1)
        z = tl.histogram(x, N)
        bias = tl.full([M, N], 1, dtype=tl.int32)
        # check that histogram produces object compatible with broadcasting
        biased = z + bias
        tl.store(z_ptr + offset2, z)

    torch.manual_seed(17)
    x = torch.randint(0, N, (M, ), device=device, dtype=torch.int32)
    z = torch.empty(N, dtype=torch.int32, device=device)
    # torch.histc does not work when the input type is not float and the device is CPU
    # https://github.com/pytorch/pytorch/issues/74236
    # This is a workload by converting the input to float
    z_torch = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    histogram_kernel[(1, )](x, z, M=M, N=N)
    assert (z_torch == z).all()


@pytest.mark.parametrize("M, N", [(1, 64), (2, 32), (4, 16), (8, 8), (16, 4), (32, 2), (64, 1)])
def test_scan_1d(M, N, device):

    @triton.jit
    def scan_kernel(out_ptr, in_ptr, M: tl.constexpr, N: tl.constexpr):
        input = tl.load(in_ptr + tl.arange(0, M))
        output = tl.cumsum(input).reshape([1, M]).broadcast_to([N, M])
        tl.store(out_ptr + tl.arange(0, M * N), output.reshape([M * N]))

    x = torch.randint(-100, 100, (M, ), dtype=torch.int32, device=device)
    output = torch.empty(M * N, dtype=torch.int32, device=device)

    scan_kernel[(1, )](output, x, M, N)

    ref = torch.cumsum(x, dim=0).reshape([1, M]).broadcast_to([N, M]).reshape([M * N])
    torch.testing.assert_close(ref.to(torch.int32), output, atol=0, rtol=0)


@pytest.mark.interpreter
@pytest.mark.parametrize("op", ['sum', 'max', 'min'])
@pytest.mark.parametrize("BLOCK_N", [32, 64, 128])
@pytest.mark.parametrize("N", [512, 1024, 2048])
@pytest.mark.parametrize("num_pid_n", [2, 4])
def test_optimize_thread_locality(op, BLOCK_N, N, num_pid_n, device):

    @triton.jit
    def kernel(X, Y, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        start_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        num_pid_n = tl.num_programs(1)
        local = INITIALIZE_PATCH
        off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        for start_n in range(pid_n, tl.cdiv(N, BLOCK_N), num_pid_n):
            off_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
            Xs = X + off_m[:, None] * N + off_n[None, :]
            x = tl.load(Xs)
            local = ACCUMULATE_PATCH
        tl.store(Y + off_m * num_pid_n + pid_n, local)

    initialize_patch = {
        'sum': 'tl.zeros([BLOCK_M], dtype=tl.float32)',
        'max': 'tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)',
        'min': 'tl.full([BLOCK_M], float("inf"), dtype=tl.float32)',
    }[op]
    reduce_patch = {
        'sum': 'local + tl.sum(x, axis=1)',
        'max': 'tl.maximum(local, tl.max(x, axis=1))',
        'min': 'tl.minimum(local, tl.min(x, axis=1))',
    }[op]
    numpy_op = {
        'sum': np.sum,
        'max': np.max,
        'min': np.min,
    }[op]
    kernel = patch_kernel(kernel, {'ACCUMULATE_PATCH': reduce_patch, 'INITIALIZE_PATCH': initialize_patch})
    torch.manual_seed(0)
    BLOCK_M = 32
    x = torch.randn((BLOCK_M, N), dtype=torch.float32, device=device)
    y = torch.randn((BLOCK_M, num_pid_n), dtype=torch.float32, device=device)
    h = kernel[(1, num_pid_n, 1)](x, y, N, BLOCK_M, BLOCK_N)
    if not is_interpreter():
        assert h.asm['ttgir'].count(
            '"tt.reduce"') == 2, "tt.reduce should be called twice, otherwise the optimization didn't work"
    y_ref = numpy_op(x.cpu().numpy(), axis=1, keepdims=True)
    y_tri = numpy_op(y.cpu().numpy(), axis=1, keepdims=True)
    np.testing.assert_allclose(y_tri, y_ref, rtol=0.01, atol=1e-3)


scan_layouts = [
    BlockedLayout([1, 4], [4, THREADS_PER_WARP // 4], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 1], [4, THREADS_PER_WARP // 4], [1, 4], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [4, THREADS_PER_WARP // 4], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [8, THREADS_PER_WARP // 8], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [4, THREADS_PER_WARP // 4], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 1], [4, THREADS_PER_WARP // 4], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [4, THREADS_PER_WARP // 4], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [8, THREADS_PER_WARP // 8], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 2], [1, THREADS_PER_WARP // 1], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
]


@pytest.mark.parametrize("M, N", [[32, 16], [32, 32], [32, 64], [64, 32]])
@pytest.mark.parametrize("src_layout", scan_layouts)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("add_overflow_check", [False, True])
def test_scan_layouts(M, N, src_layout, axis, add_overflow_check, device, tmp_path: pathlib.Path):

    overflow_check = """
        %17 = arith.extsi %arg2 : i32 to i64
        %18 = arith.extsi %arg3 : i32 to i64
        %19 = arith.addi %17, %18 : i64
        %i32.min = arith.constant -2147483648: i64
        %i32.max = arith.constant 2147483647: i64
        %20 = arith.cmpi slt, %19, %i32.max : i64
        %21 = arith.cmpi sge, %19, %i32.min : i64
        %22 = arith.andi %20, %21 : i1
        tt.assert %22, "overflow detected" : i1
    """

    ir = f"""
    #blocked = {src_layout}
    module attributes {{"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
    tt.func public @kernel_0d1d(%arg0: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
      %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #blocked>
      %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
      %1 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M}x1xi32, #blocked>
      %2 = arith.muli %1, %cst : tensor<{M}x1xi32, #blocked>
      %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<{M}x1x!tt.ptr<i32>, #blocked>
      %4 = tt.addptr %3, %2 : tensor<{M}x1x!tt.ptr<i32>, #blocked>, tensor<{M}x1xi32, #blocked>
      %5 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
      %6 = tt.expand_dims %5 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{N}xi32, #blocked>
      %7 = tt.broadcast %4 : tensor<{M}x1x!tt.ptr<i32>, #blocked> -> tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      %8 = tt.broadcast %6 : tensor<1x{N}xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
      %9 = tt.addptr %7, %8 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>, tensor<{M}x{N}xi32, #blocked>
      %10 = tt.load %9 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      %11 = "tt.scan"(%10) <{{axis = {axis} : i32, reverse = false}}> ({{
      ^bb0(%arg2: i32, %arg3: i32):
        %16 = arith.addi %arg2, %arg3 : i32{overflow_check if add_overflow_check else ""}
        tt.scan.return %16 : i32
      }}) : (tensor<{M}x{N}xi32, #blocked>) -> tensor<{M}x{N}xi32, #blocked>
      %12 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<{M}x1x!tt.ptr<i32>, #blocked>
      %13 = tt.addptr %12, %2 : tensor<{M}x1x!tt.ptr<i32>, #blocked>, tensor<{M}x1xi32, #blocked>
      %14 = tt.broadcast %13 : tensor<{M}x1x!tt.ptr<i32>, #blocked> -> tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      %15 = tt.addptr %14, %8 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>, tensor<{M}x{N}xi32, #blocked>
      tt.store %15, %11 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      tt.return
    }}
    }}
    """

    temp_file = tmp_path / "test_scan_layouts.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    rs = RandomState(17)
    x = rs.randint(-100, 100, (M, N)).astype('int32')

    z = np.zeros((M, N)).astype('int32')
    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    kernel[(1, 1, 1)](x_tri, z_tri)

    z_ref = np.cumsum(x, axis=axis)

    np.testing.assert_equal(z_ref, z_tri.cpu().numpy())


layouts = [
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    MmaLayout(version=(2, 0), warps_per_cta=[2, 4], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1],
              instr_shape=[16, 8]),
    MmaLayout(version=(3, 0), warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0],
              instr_shape=[16, 16, 16]),
    MfmaLayout(version=(2, 0), warps_per_cta=[4, 1], instr_shape=[32, 32], is_transposed=False),
    MfmaLayout(version=(2, 0), warps_per_cta=[1, 4], instr_shape=[32, 32], is_transposed=False),
    MfmaLayout(version=(2, 0), warps_per_cta=[4, 1], instr_shape=[32, 32], is_transposed=True),
    MfmaLayout(version=(2, 0), warps_per_cta=[1, 4], instr_shape=[32, 32], is_transposed=True),
    WmmaLayout(version=1, warps_per_cta=[4, 1]),
    WmmaLayout(version=1, warps_per_cta=[1, 4]),
    DotOperandLayout(parent=MmaLayout([2, 0], [2, 4], [1, 1], [1, 1], [1, 0], [16, 8]), op_idx=1, k_width=8),
    DotOperandLayout(parent=MmaLayout([3, 0], [8, 1], [1, 1], [1, 1], [1, 0], [16, 32, 16]), op_idx=0, k_width=2),
    # FIXME: Do not enable these tests until the SLPVectorizor problem with nvptx target has been resolved
    # SliceLayout(dim=1, parent=BlockedLayout([1, 4, 1], [1, 8, THREADS_PER_WARP // 8], [1, 1, 4], [2, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 2])),
    # SliceLayout(dim=0, parent=BlockedLayout([1, 4, 1], [1, 8, THREADS_PER_WARP // 8], [1, 4, 1], [2, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 2])),
    SliceLayout(dim=0, parent=MmaLayout([2, 0], [4, 1, 1], [1, 1, 1], [1, 1, 1], [2, 1, 0], [1, 16, 8])),
    SliceLayout(
        dim=1, parent=DotOperandLayout(parent=MmaLayout([2, 0], [4, 1, 1], [1, 1, 1], [1, 1, 1], [2, 1, 0], [1, 16, 8]),
                                       op_idx=1, k_width=2)),
    LinearLayout(register=[[0, 16], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane=[[0, 0], [0, 1], [0, 2], [0, 4],
                                                                                    [0, 8]], warp=[[32, 0], [0, 32]],
                 block=[]),
]


@pytest.mark.parametrize("M, N", [[128, 16], [128, 128], [32, 128], [32, 32], [16, 16]])
@pytest.mark.parametrize("src_layout", filter_layouts(layouts))
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("epilogue_kind", ['reduce1d', 'reduce2d', 'expand_reduce2d'])
@pytest.mark.parametrize("dtype_str,add_overflow_check", [("int32", False), ("int32", True), ("float32", False),
                                                          ("float16", False)])
@pytest.mark.parametrize("reduce_op", ["sum", "max"])
def test_reduce_layouts(M, N, src_layout, axis, epilogue_kind, dtype_str, add_overflow_check, reduce_op, device,
                        tmp_path: pathlib.Path):
    if isinstance(src_layout,
                  (MfmaLayout, MmaLayout)) and (M < src_layout.instr_shape[0] or N < src_layout.instr_shape[1]):
        pytest.skip("Skipping because tensor shape is smaller than M(f)maLayout instr_shape")
    if is_hip() and isinstance(src_layout, MfmaLayout) and ((M, N) == (128, 128)):
        pytest.skip("Skipping test because it runs out of shared memory")
    if reduce_op == "sum" and dtype_str == "float16" and M * N > 1024:
        pytest.skip("Skipping sum reduction on float16 due to accuracy issues")
    if is_hip() and isinstance(src_layout, LinearLayout):
        pytest.skip("FIXME: LinearLayout not supported on HIP")

    if isinstance(src_layout, MmaLayout) and src_layout.version == 3:
        src_layout.instr_shape[2] = 16 if dtype_str == "float16" else 8

    overflow_check = """
        %18 = arith.extsi %arg3 : i32 to i64
        %19 = arith.extsi %arg4 : i32 to i64
        %20 = arith.addi %18, %19 : i64
        %i32.min = arith.constant -2147483648: i64
        %i32.max = arith.constant 2147483647: i64
        %21 = arith.cmpi slt, %20, %i32.max : i64
        %22 = arith.cmpi sge, %20, %i32.min : i64
        %23 = arith.andi %21, %22 : i1
        tt.assert %23, "overflow detected" : i1
    """

    ty = {"int32": "i32", "float32": "f32", "float16": "f16"}[dtype_str]
    arith_op = {
        "max": {"int32": "arith.maxsi", "float32": "arith.maximumf", "float16": "arith.maximumf"},  #
        "sum": {"int32": "arith.addi", "float32": "arith.addf", "float16": "arith.addf"}
    }[reduce_op][dtype_str]
    numpy_op = {"max": np.max, "sum": np.sum}[reduce_op]
    rdims_1d = f"{N}" if axis == 0 else f"{M}"
    rdims_2d = f"1x{N}" if axis == 0 else f"{M}x1"
    store_range = "%7" if axis == 0 else "%1"
    warps = warps_per_cta(src_layout, [M, N])
    num_warps = int(np.prod(warps))
    blocked = BlockedLayout([1, 1], [32, THREADS_PER_WARP // 32], [4, num_warps // 4], [0, 1], [1, 1], [1, 1], [0, 1])
    one_d_layout = BlockedLayout([1], [THREADS_PER_WARP], [num_warps], [0], [1], [1], [0])

    expanded_shape = f"1x{N}" if axis == 0 else f"{M}x1"
    other_axis = 1 - axis
    epilogue = {
        "reduce1d":
        f"""
        %14 = tt.splat %arg2 : !tt.ptr<{ty}> -> tensor<{rdims_2d}x!tt.ptr<{ty}>, #blocked>
        %15 = tt.addptr %14, {store_range} : tensor<{rdims_2d}x!tt.ptr<{ty}>, #blocked>, tensor<{rdims_2d}xi32, #blocked>
        %16 = {GPU_DIALECT}.convert_layout %13 : tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #src}}>> -> tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #blocked}}>>
        %17 = tt.expand_dims %16 {{axis = {axis} : i32}} : tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #blocked}}>> -> tensor<{rdims_2d}x{ty}, #blocked>
        tt.store %15, %17 : tensor<{rdims_2d}x!tt.ptr<{ty}>, #blocked>
        tt.return
        }}
        }}
    """, "reduce2d":
        f"""
        %14 = "tt.reduce"(%13) ({{
        ^bb0(%arg3: {ty}, %arg4: {ty}):
          %17 = {arith_op} %arg3, %arg4 : {ty}{overflow_check if add_overflow_check else ""}
          tt.reduce.return %17 : {ty}
        }}) {{axis = 0 : i32}} : (tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #src}}>>) -> {ty}
        tt.store %arg2, %14 : !tt.ptr<{ty}>
        tt.return
        }}
        }}
    """, "expand_reduce2d":
        f"""
        %14 = tt.expand_dims %13 {{axis = {axis} : i32}} : tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #src}}>> -> tensor<{expanded_shape}x{ty}, #src>
        %15 = "tt.reduce"(%14) ({{
        ^bb0(%arg3: {ty}, %arg4: {ty}):
          %17 = {arith_op} %arg3, %arg4 : {ty}{overflow_check if add_overflow_check else ""}
          tt.reduce.return %17 : {ty}
        }}) {{axis = {other_axis} : i32}} : (tensor<{expanded_shape}x{ty}, #src>) -> (tensor<1x{ty}, #{GPU_DIALECT}.slice<{{dim = {other_axis}, parent = #src}}>>)
        %16 = ttg.convert_layout %15 : tensor<1x{ty}, #{GPU_DIALECT}.slice<{{dim = {other_axis}, parent = #src}}>> -> tensor<1x{ty}, #one_d_layout>
        %17 = tt.splat %arg2 : !tt.ptr<{ty}> -> tensor<1x!tt.ptr<{ty}>, #one_d_layout>
        tt.store %17, %16 : tensor<1x!tt.ptr<{ty}>, #one_d_layout>
        tt.return
        }}
        }}
                """
    }[epilogue_kind]

    ir = f"""
    #blocked = {blocked}
    #src = {src_layout}
    #one_d_layout = {one_d_layout}
    module attributes {{"ttg.num-warps" = {num_warps} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
    tt.func public @kernel_0d1d2c3d4c(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: i32 {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #blocked}}>>
        %1 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M}x1xi32, #blocked>
        %2 = tt.splat %arg1 : i32 -> tensor<{M}x1xi32, #blocked>
        %3 = arith.muli %1, %2 : tensor<{M}x1xi32, #blocked>
        %4 = tt.splat %arg0 : !tt.ptr<{ty}> -> tensor<{M}x1x!tt.ptr<{ty}>, #blocked>
        %5 = tt.addptr %4, %3 : tensor<{M}x1x!tt.ptr<{ty}>, #blocked>, tensor<{M}x1xi32, #blocked>
        %6 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #blocked}}>>
        %7 = tt.expand_dims %6 {{axis = 0 : i32}} : tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{N}xi32, #blocked>
        %8 = tt.broadcast %5 : tensor<{M}x1x!tt.ptr<{ty}>, #blocked> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #blocked>
        %9 = tt.broadcast %7 : tensor<1x{N}xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
        %10 = tt.addptr %8, %9 : tensor<{M}x{N}x!tt.ptr<{ty}>, #blocked>, tensor<{M}x{N}xi32, #blocked>
        %11 = tt.load %10 : tensor<{M}x{N}x!tt.ptr<{ty}>, #blocked>
        %12 = {GPU_DIALECT}.convert_layout %11 : tensor<{M}x{N}x{ty}, #blocked> -> tensor<{M}x{N}x{ty}, #src>
        %13 = "tt.reduce"(%12) ({{
        ^bb0(%arg3: {ty}, %arg4: {ty}):
          %17 = {arith_op} %arg3, %arg4 : {ty}{overflow_check if add_overflow_check else ""}
          tt.reduce.return %17 : {ty}
        }}) {{axis = {axis} : i32}} : (tensor<{M}x{N}x{ty}, #src>) -> tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #src}}>>
    """ + epilogue

    temp_file = tmp_path / "test_reduce_layouts.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    rs = RandomState(17)
    x = numpy_random((M, N), dtype_str=dtype_str, rs=rs, low=0, high=10)
    reduce2d = 'reduce2d' in epilogue_kind
    z_shape = (1, 1) if reduce2d else (1, N) if axis == 0 else (M, 1)
    z = np.zeros(z_shape).astype(dtype_str)

    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    pgm = kernel[(1, 1, 1)](x_tri, x_tri.stride(0), z_tri)
    z_ref = numpy_op(x) if reduce2d else numpy_op(x, axis=axis, keepdims=True)

    if dtype_str == 'float16':
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-2)
    else:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-3)


layouts = [
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    MmaLayout(version=(2, 0), warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1],
              instr_shape=[16, 8])
]


@pytest.mark.parametrize("M", [32, 64, 128, 256])
@pytest.mark.parametrize("src_layout", filter_layouts(layouts))
def test_store_op(M, src_layout, device, tmp_path: pathlib.Path):

    ir = f"""
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "{GPU_DIALECT}.num-ctas" = 1 : i32, "{GPU_DIALECT}.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
        tt.func public @kernel(%arg0: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}) {{
            %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<{M}x!tt.ptr<f32>, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %2 = tt.addptr %1, %0 : tensor<{M}x!tt.ptr<f32>, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %3 = tt.load %2 : tensor<{M}x!tt.ptr<f32>, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %4 = tt.expand_dims %3 {{axis = 1 : i32}} : tensor<{M}xf32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xf32, #src>
            %5 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %6 = tt.expand_dims %5 {{axis = 1 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xi32, #src>
            %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<{M}x1x!tt.ptr<f32>, #src>
            %8 = tt.addptr %7, %6 : tensor<{M}x1x!tt.ptr<f32>, #src>, tensor<{M}x1xi32, #src>
            tt.store %8, %4 : tensor<{M}x1x!tt.ptr<f32>, #src>
            tt.return
        }}
    }}
    """

    temp_file = tmp_path / "test_store_op.ttgir"
    temp_file.write_text(ir)
    store_kernel = triton.compile(str(temp_file))

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, 1)).astype('float32')
    y = np.zeros((M, 1), dtype='float32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)

    pgm = store_kernel[(1, 1, 1)](x_tri, y_tri)
    y_ref = x
    np.testing.assert_allclose(y_ref, y_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


layouts = [
    # TODO (lixun): Add MfmaLayout
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    MmaLayout([3, 0], [4, 1], [1, 1], [1, 1], [1, 0], [16, 32, 16]),
    MmaLayout(version=(2, 0), warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1],
              instr_shape=[16, 8]),
    DotOperandLayout(parent=MmaLayout([3, 0], [4, 1], [1, 1], [1, 1], [1, 0], [16, 32, 16]), op_idx=0, k_width=2),
    DotOperandLayout(parent=MmaLayout([2, 0], [2, 2], [1, 1], [1, 1], [1, 0], [16, 8]), op_idx=0, k_width=2),
]


@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("src_layout", filter_layouts(layouts))
@pytest.mark.parametrize("dst_layout", filter_layouts(layouts))
@pytest.mark.parametrize("src_dim", [0, 1])
@pytest.mark.parametrize("dst_dim", [0, 1])
def test_convert1d(M, src_layout, dst_layout, src_dim, dst_dim, device, tmp_path: pathlib.Path):

    ir = f"""
    #dst = {dst_layout}
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
        tt.func public @kernel(%arg0: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
            %0 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %2 = tt.addptr %0, %1 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %3 = tt.load %2 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %4 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %5 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %6 = tt.addptr %4, %5 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %7 = {GPU_DIALECT}.convert_layout %3 : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>> -> tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            tt.store %6, %7 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            tt.return
        }}
    }}
    """
    temp_file = tmp_path / "test_convert1d.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, )).astype('int32')
    y = np.zeros((M, ), dtype='int32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)
    pgm = kernel[(1, 1, 1)](x_tri, y_tri)
    y_ref = x
    np.testing.assert_allclose(y_ref, y_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("src_layout", filter_layouts(layouts))
@pytest.mark.parametrize("dst_layout", filter_layouts(layouts))
@pytest.mark.parametrize("src_dim", [0, 1])
@pytest.mark.parametrize("dst_dim", [0, 1])
def test_convert1d_bool(M, src_layout, dst_layout, src_dim, dst_dim, device, tmp_path: pathlib.Path):

    ir = f"""
    #dst = {dst_layout}
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
        tt.func public @kernel(%arg0: !tt.ptr<i8> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
            %cst = arith.constant dense<0> : tensor<{M}xi8, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %0 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<{M}x!tt.ptr<i8>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %2 = tt.addptr %0, %1 : tensor<{M}x!tt.ptr<i8>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %3 = tt.load %2 : tensor<{M}x!tt.ptr<i8>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %4 = arith.cmpi ne, %3, %cst : tensor<{M}xi8, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %6 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %7 = tt.addptr %5, %6 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %8 = {GPU_DIALECT}.convert_layout %4 : tensor<{M}xi1, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>> -> tensor<{M}xi1, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %9 = arith.extui %8 : tensor<{M}xi1, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>> to tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            tt.store %7, %9 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            tt.return
        }}
    }}
    """
    temp_file = tmp_path / "test_convert1d.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    rs = RandomState(17)
    x = rs.randint(0, 2, (M, )).astype('bool')
    y = np.zeros((M, ), dtype='int32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)
    pgm = kernel[(1, 1, 1)](x_tri, y_tri)
    y_ref = x
    np.testing.assert_allclose(y_ref, y_tri.cpu().numpy(), rtol=0, atol=0)


@triton.jit
def _welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    w2_over_w = weight_2 / new_weight
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )


layouts = [
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    # [HIP] TO DO: some tests are flaky with the layout, so turn off them for now.
    # BlockedLayout([1, 4], [1, THREADS_PER_WARP], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [THREADS_PER_WARP // 32, 32], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1])
]


@pytest.mark.parametrize("M, N", [[128, 128], [256, 128], [256, 256], [128, 256]])
@pytest.mark.parametrize("src_layout", layouts)
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("first_axis", [0, 1])
def test_chain_reduce(M, N, src_layout, op, device, first_axis, tmp_path: pathlib.Path):

    op_str = ""
    if op == "sum":
        op_str = """
        %13 = arith.addi %arg2, %arg3 : i32
        tt.reduce.return %13 : i32"""
    elif op == "max":
        op_str = """
        %13 = arith.cmpi "sgt", %arg2, %arg3 : i32
        %14 = arith.select %13, %arg2, %arg3 : i32
        tt.reduce.return %14 : i32"""
    ir = f"""
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
    tt.func public @sum_kernel_0d1d(%arg0: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
        %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #src>
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
        %1 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xi32, #src>
        %2 = arith.muli %1, %cst : tensor<{M}x1xi32, #src>
        %3 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #src}}>>
        %4 = tt.expand_dims %3 {{axis = 0 : i32}} : tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #src}}>> -> tensor<1x{N}xi32, #src>
        %5 = tt.broadcast %2 : tensor<{M}x1xi32, #src> -> tensor<{M}x{N}xi32, #src>
        %6 = tt.broadcast %4 : tensor<1x{N}xi32, #src> -> tensor<{M}x{N}xi32, #src>
        %7 = arith.addi %5, %6 : tensor<{M}x{N}xi32, #src>
        %8 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<{M}x{N}x!tt.ptr<i32>, #src>
        %9 = tt.addptr %8, %7 : tensor<{M}x{N}x!tt.ptr<i32>, #src>, tensor<{M}x{N}xi32, #src>
        %10 = tt.load %9 : tensor<{M}x{N}x!tt.ptr<i32>, #src>
        %11 = "tt.reduce"(%10) ({{
        ^bb0(%arg2: i32, %arg3: i32):
        {op_str}
        }}) {{axis = {first_axis} : i32}} : (tensor<{M}x{N}xi32, #src>) -> tensor<{M if first_axis == 1 else N}xi32, #{GPU_DIALECT}.slice<{{dim = {first_axis}, parent = #src}}>>
        %12 = "tt.reduce"(%11) ({{
        ^bb0(%arg2: i32, %arg3: i32):
        {op_str}
        }}) {{axis = 0 : i32}} : (tensor<{M if first_axis == 1 else N}xi32, #{GPU_DIALECT}.slice<{{dim = {first_axis}, parent = #src}}>>) -> i32
        tt.store %arg1, %12 : !tt.ptr<i32>
        tt.return
    }}
    }}
    """
    temp_file = tmp_path / "test_chain_reduce.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, N)).astype('int32')

    z = np.zeros((1, )).astype('int32')

    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    pgm = kernel[(1, 1, 1)](x_tri, z_tri)
    if op == "sum":
        z_ref = np.sum(x)
    elif op == "max":
        z_ref = np.max(x)

    np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


@pytest.mark.interpreter
def test_generic_reduction(device):

    @triton.jit
    def var_mean_kernel(X, out_mean, out_var, BLOCK: tl.constexpr):
        xindex = tl.arange(0, BLOCK)
        x = tl.load(X + xindex)
        mean = x
        m2 = tl.zeros_like(x)
        weight = tl.full(x.shape, 1, x.dtype)
        (mean, m2, weight) = tl.reduce((mean, m2, weight), 0, _welford_combine)
        tl.store(out_mean, mean)
        tl.store(out_var, m2 / weight)

    SIZE = 512
    x = torch.rand(SIZE, device=device)
    out_mean = torch.empty((), device=device)
    out_var = torch.empty((), device=device)

    var_mean_kernel[(1, )](x, out_mean, out_var, BLOCK=SIZE)

    expect_var, expect_mean = torch.var_mean(x, dim=0, correction=0)
    torch.testing.assert_close(out_mean, expect_mean)
    torch.testing.assert_close(out_var, expect_var)


# ---------------
# test permute
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str, shape, perm", [(dtype, shape, perm)
                                                    # TODO: bfloat16
                                                    for dtype in ['float8e4b15', 'float16', 'float32']
                                                    for shape in [(64, 64), (128, 128)]
                                                    for perm in [(1, 0)]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_permute(dtype_str, shape, perm, num_ctas, device):
    check_type_supported(dtype_str, device)  # bfloat16 on cc < 80 will not be tested
    if dtype_str == "float8e4b15" and (is_hip() or (is_cuda() and torch.cuda.get_device_capability() >= (9, 0))):
        pytest.skip("float8e4b15 not supported on ROCm or CUDA >= 9.0")
    if is_hip():
        if shape == (128, 128) and dtype_str == 'float32':
            pytest.skip("TODO Out of LDS for float32 with shape 128x128")

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xn, Z, stride_zm, stride_zn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        tl.store(Zs, tl.load(Xs))

    # input
    x = numpy_random(shape, dtype_str=dtype_str)
    # triton result
    z_tri = to_triton(np.empty_like(x), device=device, dst_type=dtype_str)
    z_tri_contiguous = to_triton(np.empty_like(x), device=device, dst_type=dtype_str)
    x_tri = to_triton(x, device=device, dst_type=dtype_str)
    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1), z_tri, z_tri.stride(1), z_tri.stride(0),
                         BLOCK_M=shape[0], BLOCK_N=shape[1], num_ctas=num_ctas)
    pgm_contiguous = kernel[(1, 1)](x_tri, x_tri.stride(1),
                                    x_tri.stride(0), z_tri_contiguous, z_tri_contiguous.stride(0),
                                    z_tri_contiguous.stride(1), BLOCK_M=shape[0], BLOCK_N=shape[1], num_ctas=num_ctas)
    # numpy result
    if dtype_str == 'float8e4b15':
        ty = tl.float8e4b15
        z_ref = serialize_fp8(deserialize_fp8(x, ty).T.copy(), ty)
        z_tri = z_tri.base
        z_tri_contiguous = z_tri_contiguous.base
    else:
        z_ref = x.transpose(*perm)
    # compare
    np.testing.assert_allclose(to_numpy(z_tri), z_ref)
    np.testing.assert_allclose(to_numpy(z_tri_contiguous), z_ref)

    if not is_cuda():
        return

    # parse ptx to make sure ld/st are vectorized
    ptx = pgm.asm['ptx']
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx
    ptx = pgm_contiguous.asm['ptx']
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ["int32", "int8"])
@pytest.mark.parametrize("shape", [(2, 4), (16, 16)])
@pytest.mark.parametrize("perm", list(itertools.permutations([0, 1])))
def test_trans_2d(dtype_str, shape, perm, device):

    @triton.jit
    def kernel(In, Out, in_shape1: tl.constexpr, in_shape2: tl.constexpr, ou_shape1: tl.constexpr,
               ou_shape2: tl.constexpr, trans1: tl.constexpr, trans2: tl.constexpr):
        in_offs = tl.arange(0, in_shape1)[:, None] * in_shape2 + tl.arange(0, in_shape2)[None, :]
        ou_offs = tl.arange(0, ou_shape1)[:, None] * ou_shape2 + tl.arange(0, ou_shape2)[None, :]
        tl.store(Out + ou_offs, tl.permute(tl.load(In + in_offs), (trans1, trans2)))

    input = torch.arange(math.prod(shape), dtype=getattr(torch, dtype_str), device=device).reshape(shape)
    expected = torch.permute(input, perm)
    # Don't do zeros_like -- that copies the layout, which we don't want.
    actual = torch.zeros(expected.shape, dtype=getattr(torch, dtype_str), device=device)

    kernel[(1, )](input, actual, *shape, *[shape[i] for i in perm], *perm)

    np.testing.assert_equal(to_numpy(expected), to_numpy(actual))


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ["int32", "int8"])
@pytest.mark.parametrize("shape", [(2, 2, 8, 64), (4, 4, 4, 4)])
@pytest.mark.parametrize("perm", list(itertools.permutations([0, 1, 2, 3])))
def test_trans_4d(dtype_str, shape, perm, device):

    @triton.jit
    def kernel(In, Out,  #
               in_shape1: tl.constexpr, in_shape2: tl.constexpr, in_shape3: tl.constexpr, in_shape4: tl.constexpr,
               ou_shape1: tl.constexpr, ou_shape2: tl.constexpr, ou_shape3: tl.constexpr, ou_shape4: tl.constexpr,
               trans1: tl.constexpr, trans2: tl.constexpr, trans3: tl.constexpr, trans4: tl.constexpr):
        in_ptr = tl.make_block_ptr(
            base=In,
            shape=(in_shape1, in_shape2, in_shape3, in_shape4),
            strides=(in_shape4 * in_shape3 * in_shape2, in_shape4 * in_shape3, in_shape4, 1),
            offsets=(0, 0, 0, 0),
            block_shape=(in_shape1, in_shape2, in_shape3, in_shape4),
            order=(3, 2, 1, 0),
        )
        out_ptr = tl.make_block_ptr(
            base=Out,
            shape=(ou_shape1, ou_shape2, ou_shape3, ou_shape4),
            strides=(ou_shape4 * ou_shape3 * ou_shape2, ou_shape4 * ou_shape3, ou_shape4, 1),
            offsets=(0, 0, 0, 0),
            block_shape=(ou_shape1, ou_shape2, ou_shape3, ou_shape4),
            order=(3, 2, 1, 0),
        )
        tl.store(out_ptr, tl.load(in_ptr).permute((trans1, trans2, trans3, trans4)))

    input = torch.arange(math.prod(shape), dtype=getattr(torch, dtype_str), device=device).reshape(shape)
    expected = torch.permute(input, perm)
    # Don't do zeros_like -- that copies the layout, which we don't want.
    actual = torch.zeros(expected.shape, dtype=getattr(torch, dtype_str), device=device)

    kernel[(1, )](input, actual, *shape, *[shape[i] for i in perm], *perm, num_warps=8)

    np.testing.assert_equal(to_numpy(expected), to_numpy(actual))


# ---------------
# test dot
# ---------------


def convert_fp8_to_fp32(x, device, dtype_str):
    if dtype_str == 'float8e4nv':
        return torch.tensor(x, device=device).view(torch.float8_e4m3fn).to(torch.float32)
    elif dtype_str == 'float8e5':
        return torch.tensor(x, device=device).view(torch.float8_e5m2).to(torch.float32)
    elif dtype_str == 'float8e4b8':
        return torch.tensor(x, device=device).view(torch.float8_e4m3fnuz).to(torch.float32)
    elif dtype_str == 'float8e5b16':
        return torch.tensor(x, device=device).view(torch.float8_e5m2fnuz).to(torch.float32)
    assert "Unsupported float8 dtype"


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
def get_test_dot_base_cases():
    return [(*shape, 4, False, False, epilogue, input_precision, in_dtype, out_dtype, 1, None)
            for shape in [(64, 64, 64), (32, 32, 32), (16, 16, 16)]
            for epilogue in ['none', 'trans', 'add-matrix', 'add-rows', 'add-cols', 'softmax', 'chain-dot']
            for input_precision in ['tf32', 'tf32x3', 'ieee']
            for in_dtype, out_dtype in [('float16', 'float16'), ('float16', 'float32'), ('float32', 'float32')]
            if not (input_precision != 'ieee' and (in_dtype in ['float16']))]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
def get_test_dot_softmax():
    return [(128, 128, 64, 8, False, False, 'softmax', 'ieee', 'float16', 'float32', 1, None)]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
def get_test_dot_mixed_sizes_cases():
    available_kpack = [1, 2 if is_hip() else 1]
    available_precision = ["tf32" if is_cuda() else "ieee"]
    return [
        (*shape_nw, col_a, col_b, 'none', input_precision, in_dtype, out_dtype, kpack, None)
        for shape_nw in [[128, 256, 32, 8], [128, 16, 32, 4], [32, 128, 64, 4], [128, 128, 64, 4], [64, 128, 128, 4],
                         [32, 128, 64, 2], [64, 64, 32, 4], [32, 32, 128, 16], [128, 128, 64, 2], [64, 128, 128, 2]]
        for input_precision in available_precision
        for col_a in [True, False]
        for col_b in [True, False]
        for in_dtype, out_dtype in [('int8', 'int8'), ('float16', 'float16'), ('float16',
                                                                               'float32'), ('float32', 'float32')]
        for kpack in available_kpack
    ]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
# introduced in #2370
def get_test_dot_transposed_op_base_cases():
    return [(64, 64, 64, 4, col_a, col_b, 'none', 'ieee', 'float32', 'float32', 1, None)
            for col_a in [True, False]
            for col_b in [True, False]]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
# Introduced in #2750
def get_test_dot_h100_shortcut_cases():
    return [(64, 64, 64, 4, False, False, 'chain-dot', 'ieee', 'bfloat16', 'float32', 1, None)]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
# introduced in #3908
def get_test_dot_mfma_edge_cases():
    if not is_hip_cdna():
        return []
    return [(16, 16, 8, 4, False, False, 'None', 'ieee', 'float32', 'float32', 1, None),
            (32, 16, 8, 4, False, False, 'None', 'ieee', 'float16', 'float16', 1, None)]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
# introduced in #3370
def get_test_dot_fp8_output_cases():
    return [(128, 128, 64, 4, False, False, 'chain-dot', 'ieee', float8_type, 'float32', 1, None)
            for float8_type in ["float8e5", "float8e4nv"]]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
# introduced in #5406
def get_test_dot_small_k_mfma_cases():
    if not is_hip_cdna():
        return []
    return [(32, 32, k_size, 4, False, False, 'None', 'ieee', in_dtype, out_dtype, 1, mma_nonk_size)
            for k_size in [1, 2, 4, 8]
            for in_dtype, out_dtype in [('float16', 'float32'), ('int8', 'int32')]
            for mma_nonk_size in mma_nonk_sizes]


# M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size
# introduced in #4516
def get_test_dot_small_mn_fma_cases():
    return [(*shape_nw, False, False, epilogue, 'ieee', in_dtype, out_dtype, 1, None)
            for shape_nw in [(2, 2, 16, 1), (1, 64, 64, 1), (64, 2, 64, 2), (64, 64, 4, 4)]
            for epilogue in ['none', 'trans', 'add-matrix', 'add-rows', 'add-cols']
            for in_dtype, out_dtype in [('float16', 'float16'), ('float32', 'float32')]]


def get_test_dot_double_rate_cases():
    if not is_hip_cdna():
        return []
    return [(32, 32, 16, 4, False, False, 'None', 'ieee', 'float16', 'float32', 1, None),
            (32, 32, 16, 4, False, False, 'None', 'ieee', 'bfloat16', 'float32', 1, None),
            (16, 16, 32, 4, False, False, 'None', 'ieee', 'float16', 'float32', 1, None),
            (16, 16, 32, 4, False, False, 'None', 'ieee', 'bfloat16', 'float32', 1, None)]


@pytest.mark.interpreter
@pytest.mark.parametrize(
    "M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size",
    get_test_dot_double_rate_cases() + \
    get_test_dot_base_cases() + \
    get_test_dot_mixed_sizes_cases() + \
    get_test_dot_transposed_op_base_cases() + \
    get_test_dot_h100_shortcut_cases() + \
    get_test_dot_mfma_edge_cases() + \
    get_test_dot_fp8_output_cases() + \
    get_test_dot_small_k_mfma_cases() + \
    get_test_dot_small_mn_fma_cases() + \
    get_test_dot_softmax())
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_dot(M, N, K, num_warps, col_a, col_b, epilogue, input_precision, in_dtype, out_dtype, kpack, mma_nonk_size,
             num_ctas, device):
    if is_interpreter():
        if in_dtype == 'bfloat16':
            pytest.skip("bfloat16 is not supported in the interpreter")
    else:
        if not is_hip() and (M < 16 or N < 16 or K < 16):
            pytest.skip("small dots are supported only on HIP at the moment")
        if is_cuda():
            capability = torch.cuda.get_device_capability()

            if capability[0] < 7:
                pytest.skip("Only test tl.dot() on devices with sm >= 70")
            if capability[0] < 8:
                if capability[1] == 0 and in_dtype == 'int8':
                    pytest.skip("Only test int8 on devices with sm >= 75")
                if input_precision != "ieee":
                    pytest.skip("Only test tf32 on devices with sm >= 80")
            if capability[0] == 7:
                if (M, N, K, num_warps) in [(128, 256, 32, 8), (64, 128, 128, 4), (64, 128, 128, 2)]:
                    pytest.skip("shared memory out of resource")
                if out_dtype == 'float16':
                    # TODO: support out_dtype=float16 for tl.dot on V100
                    pytest.skip("Only test out_dtype=float16 on devices with sm >=80")
            if capability[0] < 9 and in_dtype == 'float8e4nv':
                pytest.skip("float8e4nv not supported on sm <= 80")

        if is_hip():
            if in_dtype in ("float8e5", "float8e4nv") and not is_hip_cdna4():
                pytest.skip(f"{in_dtype} only supported on CDNA4")
            if in_dtype in ("float8e5b16", "float8e4b8") and not is_hip_cdna3():
                pytest.skip(f"{in_dtype} only supported on CDNA3")
            if not ((input_precision == "ieee") or (input_precision == "tf32" and is_hip_cdna3())):
                pytest.skip(f"{input_precision} not supported on HIP")
            if kpack == 2 and in_dtype == 'int8' and K < 64:
                pytest.skip("kpack too large for K")
        if not is_hip() and kpack == 2:
            pytest.skip("Skip duplicated tests on nv path")

    torch.backends.cuda.matmul.allow_tf32 = input_precision == "tf32"

    if num_ctas > 1 and in_dtype == 'int8':
        # FIXME: mma v2 with num_ctas > 1 does not work
        pytest.skip()
    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xk, Y, stride_yk, stride_yn, W, stride_wn, stride_wl, Z, stride_zm, stride_zn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ADD_MATRIX: tl.constexpr,
               ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr, INPUT_PRECISION: tl.constexpr, DO_SOFTMAX: tl.constexpr,
               CHAIN_DOT: tl.constexpr, COL_A: tl.constexpr, COL_B: tl.constexpr, out_dtype: tl.constexpr = tl.float32):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_l = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Ws = W + off_n[:, None] * stride_wn + off_l[None, :] * stride_wl
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        x = tl.load(Xs)
        y = tl.load(Ys)
        z = tl.dot(x, y, input_precision=INPUT_PRECISION, out_dtype=out_dtype)
        if ADD_MATRIX:
            z += tl.load(Zs)
        if ADD_ROWS:
            ZRs = Z + off_m * stride_zm
            z += tl.load(ZRs)[:, None]
        if ADD_COLS:
            ZCs = Z + off_n * stride_zn
            z += tl.load(ZCs)[None, :]
        if DO_SOFTMAX:
            max = tl.max(z, 1)
            z = z - max[:, None]
            num = tl.exp(z.to(tl.float32)).to(max.dtype)
            den = tl.sum(num, 1)
            z = num / den[:, None]
        if CHAIN_DOT:
            w = tl.load(Ws)
            z = tl.dot(z.to(w.dtype), w, input_precision=INPUT_PRECISION, out_dtype=out_dtype)
        tl.store(Zs, z)

    # input
    rs = RandomState(17)
    if col_a:
        x = numpy_random((K, M), dtype_str=in_dtype, rs=rs).T
    else:
        x = numpy_random((M, K), dtype_str=in_dtype, rs=rs)
    if col_b:
        y = numpy_random((N, K), dtype_str=in_dtype, rs=rs).T
    else:
        y = numpy_random((K, N), dtype_str=in_dtype, rs=rs)
    w = numpy_random((N, N), dtype_str=in_dtype, rs=rs)
    if 'int' not in in_dtype and 'float8' not in in_dtype:
        x *= .1
        y *= .1
    if in_dtype == 'float32' and input_precision == "tf32":
        x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')
        y = (y.view('uint32') & np.uint32(0xffffe000)).view('float32')
        w = (w.view('uint32') & np.uint32(0xffffe000)).view('float32')
    x_tri = to_triton(x, device=device, dst_type=in_dtype)
    y_tri = to_triton(y, device=device, dst_type=in_dtype)
    w_tri = to_triton(w, device=device, dst_type=in_dtype)
    # triton result
    if out_dtype == 'int8':
        z = 1 + numpy_random((M, N), dtype_str='int32', rs=rs)
    else:
        z = 1 + numpy_random((M, N), dtype_str=in_dtype, rs=rs) * .1

    z_tri = to_triton(z, device=device)
    if epilogue == 'trans':
        z_tri = torch.as_strided(z_tri, (M, N), [1, M])

    if out_dtype == 'int8':
        out_dtype = tl.int8
    elif out_dtype == 'float16' and epilogue != 'softmax':
        # TODO: for out_dtype == 'float16' and epilogue == 'softmax', it will
        # fail with the following error: 'llvm.fmul' op requires the same type
        # for all operands and results
        out_dtype = tl.float16
    else:
        out_dtype = tl.float32

    kern_kwargs = {
        'COL_A': col_a, 'COL_B': col_b, 'BLOCK_M': M, 'BLOCK_K': K, 'BLOCK_N': N, 'ADD_MATRIX':
        epilogue == 'add-matrix', 'ADD_ROWS': epilogue == 'add-rows', 'ADD_COLS': epilogue == 'add-cols', 'DO_SOFTMAX':
        epilogue == 'softmax', 'CHAIN_DOT': epilogue == 'chain-dot', 'INPUT_PRECISION': input_precision, 'num_warps':
        num_warps, 'num_ctas': num_ctas, 'out_dtype': out_dtype
    }

    if is_hip():
        kern_kwargs['kpack'] = kpack
        if mma_nonk_size is not None:
            kern_kwargs['matrix_instr_nonkdim'] = mma_nonk_size

    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1), y_tri, y_tri.stride(0), y_tri.stride(1), w_tri,
                         w_tri.stride(0), w_tri.stride(1), z_tri, z_tri.stride(0), z_tri.stride(1), **kern_kwargs)

    # torch result
    if in_dtype == 'int8':
        z_ref = np.matmul(x.astype(np.float32), y.astype(np.float32())).astype(np.int32)
    elif 'float8' in in_dtype:
        x = convert_fp8_to_fp32(x, device, in_dtype)
        y = convert_fp8_to_fp32(y, device, in_dtype)
        z_ref = to_numpy(torch.matmul(x, y))
    else:
        z_ref = np.matmul(x, y)

    if epilogue == 'add-matrix':
        z_ref += z
    if epilogue == 'add-rows':
        z_ref += z[:, 0][:, None]
    if epilogue == 'add-cols':
        z_ref += z[0, :][None, :]
    if epilogue == 'softmax':
        num = np.exp(z_ref - np.max(z_ref, axis=-1, keepdims=True))
        denom = np.sum(num, axis=-1, keepdims=True)
        z_ref = num / denom
    if epilogue == 'chain-dot':
        if 'float8' in in_dtype:
            # Reduce z_ref's precision to fp8 to match the kernel behavior
            if in_dtype == 'float8e4nv':
                z_fp8 = torch.tensor(z_ref, dtype=torch.float8_e4m3fn)
            elif in_dtype == 'float8e5':
                z_fp8 = torch.tensor(z_ref, dtype=torch.float8_e5m2)
            elif in_dtype == 'float8e4b8':
                z_fp8 = torch.tensor(z_ref, dtype=torch.float8_e4m3fnuz)
            elif in_dtype == 'float8e5b16':
                z_fp8 = torch.tensor(z_ref, dtype=torch.float8_e5m2fnuz)
            else:
                assert "Unsupported float8 dtype"
            z_ref = to_numpy(z_fp8.to(torch.float32))
            w = to_numpy(convert_fp8_to_fp32(w, device, in_dtype))
        z_ref = np.matmul(z_ref, w)
    # compare
    if in_dtype == 'float32':
        # XXX: Somehow there's a larger difference when we use float32
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-3)
    elif out_dtype == tl.float16 or in_dtype == 'bfloat16':
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-2)
    else:
        # added atol, to loose precision for float16xfloat16->float32 case
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-3)
    if not is_cuda():
        return
    # make sure ld/st are vectorized
    ptx = pgm.asm['ptx']
    if (K > 16 or N > 16 or M > 16) and (M * N // (num_warps * 32) >= 4):
        # XXX: skip small sizes because they are not vectorized
        assert 'ld.global.v4' in ptx
        if 'float8' in in_dtype:
            assert 'st.global.v2' in ptx
        else:
            assert 'st.global.v4' in ptx

    is_tcgen5 = (capability[0] == 10) and (num_warps % 4) == 0 and (M % 64) == 0 and (N % 8) == 0

    if in_dtype == 'float32' and input_precision != "ieee":
        if is_tcgen5:
            assert re.search(r'tcgen05.mma.cta_group::1.kind::tf32', ptx)
        else:
            assert re.search(r'[mma|wgmma.mma_async].sync.aligned.m\d+n\d+k8(?:.row.col)?.f32.tf32.tf32', ptx)
    elif in_dtype == 'float16' and out_dtype == tl.float32:
        if is_tcgen5:
            assert re.search(r'tcgen05.mma.cta_group::1.kind::f16', ptx)
        elif capability[0] == 7 and capability[1] == 5:  # Turing
            assert re.search(r'mma.sync.aligned.m\d+n\d+k8(?:.row.col)?.f32.f16.f16', ptx)
        else:
            assert re.search(r'[mma|wgmma.mma_async].sync.aligned.m\d+n\d+k16(?:.row.col)?.f32.f16.f16', ptx)
    elif in_dtype == 'float16' and out_dtype == tl.float16:
        if is_tcgen5:
            assert re.search(r'tcgen05.mma.cta_group::1.kind::f16', ptx)
        elif capability[0] == 7 and capability[1] == 5:  # Turing
            assert re.search(r'mma.sync.aligned.m\d+n\d+k8(?:.row.col)?.f16.f16.f16', ptx)
        else:
            assert re.search(r'[mma|wgmma.mma_async].sync.aligned.m\d+n\d+k16(?:.row.col)?.f16.f16.f16', ptx)
    elif in_dtype == 'int8':
        if capability[0] == 7 and capability[1] == 5:  # Turing
            assert 'mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32' in ptx
        else:
            assert 'wgmma.mma_async.sync.aligned' in ptx or\
                'mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32' in ptx
    elif in_dtype == "float8e5" and out_dtype == tl.float32:
        if capability[0] == 9:
            assert 'wgmma.mma_async.sync.aligned.m64n128k32.f32.e5m2.e5m2' in ptx
    elif in_dtype == "float8e4nv" and out_dtype == tl.float32:
        if capability[0] == 9:
            assert 'wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3' in ptx
    if is_tcgen5 and epilogue == 'softmax' and M >= 128:
        # check that there is no shared memory exchange in the softmax
        pattern = (r'tcgen05\.ld\.sync\.aligned\.16x32bx2\.x64\.b32'
                   r'(?:(?!st\.shared).)*'
                   r'cvt\.rn\.f16x2\.f32')
        assert re.search(pattern, ptx, flags=re.DOTALL)


@pytest.mark.parametrize("M, N, K, col_a, col_b, rhs_scale, mxfp_type, normal_type, num_warps, mma, kpack",
                         [(M, N, K, col_a, col_b, rhs_scale, mxfp_type, normal_type, 4, mma, kpack)
                          for M, N, K in itertools.product([32, 64, 128], [32, 64, 128], [64, 128])
                          for col_a, col_b in itertools.product([True, False], repeat=2)
                          for rhs_scale in [False, True]
                          for mxfp_type in ["e2m1", "e4m3", "e5m2"]
                          for normal_type in ["e4m3", "e5m2", "bf16", "fp16"]
                          for mma in (mma_nonk_sizes if is_hip() else [16])
                          for kpack in ([1, 2] if is_hip() else [1])])
def test_scaled_dot(M, N, K, col_a, col_b, rhs_scale, mxfp_type, normal_type, num_warps, mma, kpack, device):
    if is_cuda():
        cc = torch.cuda.get_device_capability()
        if cc < (8, 9):
            pytest.skip("float8e4nv not supported on CUDA < 8.9")
    if is_hip():
        if not is_hip_cdna():
            pytest.skip("scaled_dot only implemented for HIP CDNA")
        if "e4m3" in (mxfp_type, normal_type):
            if not (is_hip_cdna3() or is_hip_cdna4()):
                pytest.skip(f"scaled_dot({mxfp_type}, {normal_type}) only implemented for CDNA3 and CDNA4")
        if mma == 16 and K == 64:
            pytest.skip(f"K == {K} too small for mfma {mma} in scaled_dot")

    @triton.jit
    def dot_scale_kernel(a_base, stride_a0, stride_a1, a_scale, b_base, stride_b0, stride_b1, b_scale, out,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, type_a: tl.constexpr,
                         type_b: tl.constexpr):
        DIV_FACTOR_A: tl.constexpr = 2 if type_a == "e2m1" else 1
        DIV_FACTOR_B: tl.constexpr = 2 if type_b == "e2m1" else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
        a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + tl.arange(0,
                                                                                PACKED_BLOCK_K_A)[None, :] * stride_a1
        b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * stride_b0 + tl.arange(0,
                                                                                         BLOCK_N)[None, :] * stride_b1

        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
        if a_scale is not None:
            scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0,
                                                                                               SCALE_BLOCK_K)[None, :]
            a_scale = tl.load(scale_a_ptr)
        if b_scale is not None:
            scale_b_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + tl.arange(0,
                                                                                               SCALE_BLOCK_K)[None, :]
            b_scale = tl.load(scale_b_ptr)
        c = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b)
        out_ptr = out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        tl.store(out_ptr, c.to(tl.bfloat16))

    @triton.jit
    def mxfp_upcast_kernel(
        x_ptr,
        scale_ptr,
        mxfp_ptr,
        N,
        e_bits: tl.constexpr,
        m_bits: tl.constexpr,
        to_type: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # x.shape ==     (N, 32) for fp8 or (N, 16) for fp4
        # scale.shape == (N,)
        # out.shape   == (N, 32)
        is_fp8: tl.constexpr = e_bits + m_bits == 7
        # fp8: BLOCK_SIZE -> BLOCK_SIZE // 32, 32
        # fp4: BLOCK_SIZE // 2 -> BLOCK_SIZE // 32 , 16
        PARALLEL_DIM: tl.constexpr = BLOCK_SIZE // 32
        LAST_DIM: tl.constexpr = 32 if is_fp8 else 16
        LOAD_SIZE: tl.constexpr = LAST_DIM * PARALLEL_DIM

        offsets = (tl.program_id(0) * LOAD_SIZE + tl.arange(0, PARALLEL_DIM)[:, None] * LAST_DIM +
                   tl.arange(0, LAST_DIM)[None, :])
        x = tl.load(x_ptr + offsets, mask=offsets < N * LAST_DIM)

        offsets = tl.program_id(0) * PARALLEL_DIM + tl.arange(0, PARALLEL_DIM)[:, None]
        scale = tl.load(scale_ptr + offsets, mask=offsets < N)
        tl.static_assert(scale.dtype == tl.uint8)
        tl.static_assert(x.dtype == tl.uint8)

        if to_type == tl.bfloat16:
            upcasted_scale = (scale.to(tl.uint16) << 7).to(tl.bfloat16, bitcast=True)
        else:
            tl.static_assert(to_type == tl.float16)
            scale_fp32 = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
            upcasted_scale = scale_fp32.to(tl.float16)

        to_e_bits: tl.constexpr = 8 if to_type == tl.bfloat16 else 5
        to_m_bits: tl.constexpr = 7 if to_type == tl.bfloat16 else 10
        if is_fp8:
            if e_bits == 5 and m_bits == 2:
                x_f8 = x.to(tl.float8e5, bitcast=True)
                upcasted_x = x_f8.to(to_type)
                # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
                non_finite_mask: tl.constexpr = ((1 << e_bits) - 1) << m_bits
                non_finite_mask_16bit: tl.constexpr = ((1 << to_e_bits) - 1) << to_m_bits
                upcasted_x = tl.where(
                    x & non_finite_mask == non_finite_mask,
                    (upcasted_x.to(tl.uint16, bitcast=True) | non_finite_mask_16bit).to(to_type, bitcast=True),
                    upcasted_x,
                )
            else:
                tl.static_assert(e_bits == 4 and m_bits == 3)
                x_f8 = x.to(tl.float8e4nv, bitcast=True)
                upcasted_x = x_f8.to(to_type)
        else:
            to_bias: tl.constexpr = 127 if to_type == tl.bfloat16 else 15
            to_point5: tl.constexpr = 16128 if to_type == tl.bfloat16 else 0x3800
            # e2m1
            em0 = x & 0x7
            em1 = x & 0x70
            x0 = (em0.to(tl.uint16) << (to_m_bits - 1)) | ((x & 0x8).to(tl.uint16) << 12)
            x1 = (em1.to(tl.uint16) << (to_m_bits - 1 - 4)) | ((x & 0x80).to(tl.uint16) << 8)
            # Three cases:
            # 1) x is normal and non-zero: Correct bias
            x0 = tl.where((em0 & 0x6) != 0, x0 + ((to_bias - 1) << to_m_bits), x0)
            x1 = tl.where((em1 & 0x60) != 0, x1 + ((to_bias - 1) << to_m_bits), x1)
            # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in bf16
            x0 = tl.where(em0 == 0x1, to_point5 | (x0 & 0x8000), x0)
            x1 = tl.where(em1 == 0x10, to_point5 | (x1 & 0x8000), x1)
            # 3) x is zero, do nothing
            upcasted_x = tl.interleave(x0, x1).to(to_type, bitcast=True)
        # Multiplication preserves infs and NaNs in upcasted_x
        mxfp = upcasted_x * upcasted_scale
        # If scale is NaN, we encode it as an inf, so we need to correct for that
        mxfp = tl.where(scale == 0xFF, float("nan"), mxfp)

        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(mxfp_ptr + offsets, tl.ravel(mxfp), mask=offsets < N * 32)

    def dot_scale_ref(x, scale_x, y, scale_y, type_x, type_y):

        def upcast(v, scale, type, comp_dtype, transposed):
            if scale is None:
                type = {
                    "e4m3": torch.float8_e4m3fn,
                    "e5m2": torch.float8_e5m2,
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                }[type]
                return v.view(type).to(comp_dtype)
            e_bits, m_bits = {"e2m1": (2, 1), "e4m3": (4, 3), "e5m2": (5, 2)}[type]
            # Packing is always on the K dimension so we transpose before upcasting then transpose back.
            if transposed:
                v = v.mT.contiguous()
            v = v.contiguous()
            v_upcast = v.new_empty(scale.shape[:-1] + (32 * scale.shape[-1], ), dtype=comp_dtype)
            N = v_upcast.numel()
            BLOCK_SIZE = 512
            grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, )
            comp_dtype = tl.float16 if comp_dtype == torch.float16 else tl.bfloat16
            mxfp_upcast_kernel[grid](v, scale, v_upcast, scale.numel(), e_bits, m_bits, comp_dtype, BLOCK_SIZE,
                                     num_warps=num_warps)
            assert v_upcast.isfinite().all()
            if transposed:
                v_upcast = v_upcast.mT
            return v_upcast

        # Upcast to fp16 if one of the input is fp16
        comp_dtype = torch.float16 if "fp16" in (type_x, type_y) else torch.bfloat16

        x_upcast = upcast(x, scale_x, type_x, comp_dtype, False)
        y_upcast = upcast(y, scale_y, type_y, comp_dtype, True)

        class AccumulateInFp32:

            def __enter__(self):
                self.prev_value = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

            def __exit__(self, exc_type, exc_val, exc_tb):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = self.prev_value

        with AccumulateInFp32():
            return torch.matmul(x_upcast, y_upcast)

    comp_dtype = torch.float16 if normal_type == "fp16" else torch.bfloat16
    # The max exponent we use to initialize data in the x/y and associated scale tensor to avoid
    # overflow when scaling.
    comp_dtype_max_exp = 6 if normal_type == "fp16" else 15

    torch.manual_seed(0)

    def make_arg(shape, ty, col_major=False):
        if col_major:
            shape = shape[:-2] + (shape[-1], shape[-2])
        if ty == "bf16" or ty == "fp16":
            ret = torch.randn(shape, dtype=comp_dtype, device=device)
            # Clamp to avoid relative error issues
            ret.clamp_(-2**comp_dtype_max_exp, 2**comp_dtype_max_exp - 1)
        else:
            if is_hip_cdna4():
                # On other chips, the A/B operands are upcasted to fp16/bf16
                # before matmul, which has larger range to avoid overflow.
                # On CDNA4, we use the V_MFMA_*_F8F6F4 instructions to
                # directly calculate matmul on F8F6F4 data. So we need
                # to narrow down the range of input to avoid overflow.
                ret = torch.randint(20, 40, shape, dtype=torch.uint8, device=device)
            else:
                ret = torch.randint(256, shape, dtype=torch.uint8, device=device)
        if col_major:
            ret = ret.mT
        return ret

    type_a = normal_type if rhs_scale else mxfp_type
    type_b = mxfp_type if rhs_scale else normal_type

    DIV_FACTOR_A = 2 if type_a == "e2m1" else 1
    DIV_FACTOR_B = 2 if type_b == "e2m1" else 1
    x = make_arg((M, K // DIV_FACTOR_A), type_a, col_major=col_a)
    y = make_arg((K // DIV_FACTOR_B, N), type_b, col_major=col_b)

    min_scale, max_scale = (0, 142) if comp_dtype == torch.bfloat16 else (124, 131)
    scale_x = torch.randint(min_scale, max_scale + 1, (M, K // 32), dtype=torch.uint8, device=device)
    scale_y = torch.randint(min_scale, max_scale + 1, (N, K // 32), dtype=torch.uint8, device=device)
    if rhs_scale:
        scale_x = None
    else:
        scale_y = None

    def make_finite(x, dtype):
        # e5m2 has too many non-finite values when sampled uniformly (1 / 32) and
        # Fp8E5M2_to_Bf16 doesn't preserve NaNs (fixme)
        if dtype not in ("e5m2", "e4m3"):
            return x
        if dtype == "e5m2" and comp_dtype == torch.float16:
            x = x & 0xB
        mask = 0x7C if dtype == "e5m2" else 0x7F
        finite = torch.arange(x.numel(), device=device, dtype=torch.uint8).reshape_as(x) % mask
        x_finite = torch.where(x & mask == mask, finite | (0x80 & x), x)
        x.copy_(x_finite)
        return x

    x = make_finite(x, type_a)
    y = make_finite(y, type_b)
    kernel_kwargs = {"num_warps": num_warps}
    if is_hip():
        kernel_kwargs["kpack"] = kpack
        kernel_kwargs["matrix_instr_nonkdim"] = mma
    z = x.new_empty((M, N), dtype=comp_dtype)
    pgm = dot_scale_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, M, N, K, type_a, type_b,
                                  **kernel_kwargs)
    z_ref = dot_scale_ref(x, scale_x, y, scale_y, type_a, type_b)
    # Bigger tolerance for AMD CDNA2 devices.
    # CDNA2 devices use reduced precision fp16 and bf16 and flush input and output denormal values
    # to zero. Detailed info is at:
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    atol = 2e-4 if is_hip_cdna2() else 1e-5
    rtol = 2e-2 if is_hip_cdna2() else 1e-2
    torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)

    # make sure ld/st are vectorized
    if is_cuda():
        ptx = pgm.asm['ptx']
        if (max(M, N) * K) // (num_warps * 32) >= 4:
            assert 'ld.global.v4' in ptx
        if M * N // (num_warps * 32) >= 4:
            assert 'st.global.v4' in ptx
        assert (re.search(r'(mma|wgmma.mma_async).sync.aligned.m\d+n\d+k16(?:.row.col)?.f32.(f|bf)16.(f|bf)16', ptx)
                or "tcgen05.mma.cta_group::1.kind::f16" in ptx)


@pytest.mark.interpreter
@pytest.mark.parametrize("B, num_warps, M, N, K, BLOCK_M, BLOCK_N, in_dtype_str, out_dtype_str",
                         [(B, num_warps, M, N, K, BLOCK_M, BLOCK_N, in_dtype_str, out_dtype_str)
                          for B in [1, 2, 4, 8]
                          for num_warps in [1, 2, 4, 8, 16]
                          for BLOCK_M, BLOCK_N in [(32, 32)]
                          for M, N, K in [(64, 64, 64), (32, 32, 32)]
                          for in_dtype_str, out_dtype_str in [('int8', 'int8'), ('float16', 'float16'),
                                                              ('float16', 'float32'), ('float32', 'float32')]] +
                         # Large block sizes
                         [(4, 4, 128, 128, 64, 64, 64, 'float16', 'float16')] +
                         # Small block sizes
                         [(B, num_warps, M, N, K, BLOCK_M, BLOCK_N, in_dtype_str, out_dtype_str)
                          for B in [1, 2, 8]
                          for num_warps in [1, 2, 4]
                          for BLOCK_M, BLOCK_N in [(1, 32), (32, 2), (8, 8)]
                          for M, N, K in [(32, 32, 32)]
                          for in_dtype_str, out_dtype_str in [('float16', 'float16'), ('float32', 'float32')]])
def test_dot3d(B, num_warps, M, N, K, BLOCK_M, BLOCK_N, in_dtype_str, out_dtype_str, device):
    if is_hip():
        # hip does not support tf32 precision, so use ieee for all tests
        input_precision = "ieee"
        arch = triton.runtime.driver.active.get_current_target().arch
        if "gfx11" in arch or "gfx12" in arch:
            if in_dtype_str == "float32":
                pytest.skip(f"{in_dtype_str} is not supported in WMMA dot, FMA does not support dot3d")
            if out_dtype_str == "float16":
                pytest.skip(f"{out_dtype_str} has low precision in WMMA dot")
    else:
        input_precision = "tf32" if is_cuda() and in_dtype_str == 'float32' else "ieee"
        if not is_interpreter() and (BLOCK_M < 16 or BLOCK_N < 16):
            pytest.skip("small dots are supported only on HIP at the moment")

    if B == 8 and M == 64 and in_dtype_str == "float32" and out_dtype_str == "float32":
        if not is_interpreter() and triton.runtime.driver.active.utils.get_device_properties(
                triton.runtime.driver.active.get_current_device())["max_shared_mem"] < 131072:
            pytest.skip(
                "Skipping tests with B = 8, M = 64, in_type = float32, out_type = float32 due to insufficient shared memory (less than 128 KB per SM) on this GPU."
            )

    @triton.jit
    def kernel(
        q_ptr,
        k_ptr,
        o_ptr,
        stride_qb,
        stride_qm,
        stride_qk,
        stride_kb,
        stride_kk,
        stride_kn,
        stride_ob,
        stride_om,
        stride_on,
        BLOCK_B: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        out_dtype: tl.constexpr = tl.float32,
    ):
        startm = tl.program_id(0) * BLOCK_M
        startn = tl.program_id(1) * BLOCK_N
        offs_b = tl.arange(0, BLOCK_B)
        offs_m = startm + tl.arange(0, BLOCK_M)
        offs_n = startn + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        q_ptrs = q_ptr + offs_b[:, None, None] * stride_qb + offs_m[None, :, None] * stride_qm + offs_k[
            None, None, :] * stride_qk
        k_ptrs = k_ptr + offs_b[:, None, None] * stride_kb + offs_k[None, :, None] * stride_kk + offs_n[
            None, None, :] * stride_kn
        q = tl.load(q_ptrs)
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k, input_precision=INPUT_PRECISION, out_dtype=out_dtype)
        o_ptrs = o_ptr + offs_b[:, None, None] * stride_ob + offs_m[None, :, None] * stride_om + offs_n[
            None, None, :] * stride_on
        tl.store(o_ptrs, qk)

    if out_dtype_str == 'int8':
        out_dtype = tl.int8
    elif out_dtype_str == 'float16':
        out_dtype = tl.float16
    else:
        out_dtype = tl.float32

    rs = RandomState(17)
    x = numpy_random((B, M, K), dtype_str=in_dtype_str, rs=rs)
    y = numpy_random((B, K, N), dtype_str=in_dtype_str, rs=rs)
    if in_dtype_str == 'int8':
        out = numpy_random((B, M, N), dtype_str='int32', rs=rs)
    else:
        if is_hip() and (BLOCK_M < 16 or BLOCK_N < 16) and out_dtype_str == 'float16':
            # float16 accumulator in FMA dot loose precision too fast
            x *= 0.1
            y *= 0.1
        out = numpy_random((B, M, N), dtype_str=out_dtype_str, rs=rs)

    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    out_tri = to_triton(out, device=device)

    BLOCK_B = B
    BLOCK_K = K

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    kernel[grid](
        x_tri,
        y_tri,
        out_tri,
        x_tri.stride(0),
        x_tri.stride(1),
        x_tri.stride(2),
        y_tri.stride(0),
        y_tri.stride(1),
        y_tri.stride(2),
        out_tri.stride(0),
        out_tri.stride(1),
        out_tri.stride(2),
        BLOCK_B=BLOCK_B,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        INPUT_PRECISION=input_precision,
        out_dtype=out_dtype,
        num_warps=num_warps,
    )

    if in_dtype_str == 'int8':
        out_ref = np.matmul(x.astype(np.float32), y.astype(np.float32)).astype(np.int32)
    else:
        out_ref = np.matmul(x, y)
    np.testing.assert_allclose(out_ref, to_numpy(out_tri), rtol=0.01, atol=1e-2)


@pytest.mark.parametrize('in_dtype', ['float32'])
def test_dot_mulbroadcasted(in_dtype, device):
    if is_cuda():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            pytest.skip("Requires sm >= 80 to run")

    @triton.jit
    def kernel(Z, X, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr):
        pidn = tl.program_id(1)
        pidm = tl.program_id(0)
        offm = tl.arange(0, BM)[:, None]
        offn = tl.arange(0, BN)[None, :]
        offak = tl.arange(0, BK)[None, :]
        offbk = tl.arange(0, BK)[:, None]
        acc = tl.full((BM, BN), 0.0, tl.float32)
        for ridx5 in range(0, K // BK):
            x = tl.load(X + ((pidm * K * BM) + (offm * K) + (ridx5 * BK) + offak))
            y = tl.load(Y + ((pidn * BN) + (offbk * N) + (ridx5 * N * BK) + offn))
            x = tl.expand_dims(x, axis=2)
            y = tl.expand_dims(y, axis=0)
            t = tl.sum(x * y, axis=1)
            acc = t + acc
        tl.store(Z + ((pidm * BM * N) + (pidn * BN) + (offm * N) + offn), acc)

    M, N, K = 256, 192, 160
    BM, BN, BK = 128, 32, 32
    rs = RandomState(17)
    x = numpy_random((M, K), dtype_str=in_dtype, rs=rs)
    y = numpy_random((K, N), dtype_str=in_dtype, rs=rs)
    x = x * 0.1
    y = y * 0.1
    z = numpy_random((M, N), dtype_str=in_dtype, rs=rs)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    z_tri = to_triton(z, device=device)
    grid = M // BM, N // BN
    h = kernel[grid](z_tri, x_tri, y_tri, M, N, K, BM, BN, BK)
    z_ref = np.matmul(x, y)
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), atol=0.01)

    if not is_cuda():
        return
    assert "tt.dot" in h.asm['ttir']
    assert re.search(r"ttg.async_wait %.* {num = 2 : i32}", h.asm["ttgir"]) is not None


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", int_dtypes + uint_dtypes + float_dtypes + ['bfloat16'])
@pytest.mark.parametrize("shape", [(), (1, ), (128, )])
def test_full(dtype_str, shape, device):
    if dtype_str in uint_dtypes and not hasattr(torch, dtype_str):
        # PyTorch only has unsigned 8, but not 16, 32, or 64
        dtype = getattr(torch, dtype_str[1:])  # uintx -> intx
    else:
        dtype = getattr(torch, dtype_str)
    check_type_supported(dtype, device)  # bfloat16 on cc < 80 will not be tested

    @triton.jit
    def kernel_static(out):
        a = GENERATE_TEST_HERE
        tl.static_assert(a.shape == SHAPE)
        out_ptr = out + tl.arange(0, 128)[:]
        tl.store(out_ptr, a)

    @triton.jit
    def kernel_dynamic(out, val, dtype: tl.constexpr):
        a = tl.full(SHAPE, val, dtype)
        tl.static_assert(a.shape == SHAPE)
        out_ptr = out + tl.arange(0, 128)[:]
        tl.store(out_ptr, a)

    kernel_static_patched = patch_kernel(kernel_static, {
        'GENERATE_TEST_HERE': f"tl.full({shape}, 2, tl.{dtype_str})",
        'SHAPE': str(list(shape)),
    })
    out_static = torch.zeros((128), dtype=dtype, device=device)
    kernel_static_patched[(1, )](out_static)
    assert torch.all(out_static == 2)

    kernel_dynamic_patched = patch_kernel(kernel_dynamic, {'SHAPE': str(list(shape))})
    out_dynamic = torch.zeros((128), dtype=dtype, device=device)
    kernel_dynamic_patched[(1, )](out_dynamic, 2, getattr(triton.language, dtype_str))
    assert torch.all(out_dynamic == 2)


@pytest.mark.parametrize("literal, dtype_str", [(1e+50, "f64"), (1e+10, "f32"), (1.0, "f32"), ('float("inf")', "f32"),
                                                ('float("-inf")', "f32"), ('float("nan")', "f32"),
                                                ('float("-nan")', "f32"), (0., "f32"), (5, "i32"), (2**40, "i64")])
def test_constexpr(literal, dtype_str, device):

    @triton.jit
    def kernel(out_ptr):
        val = GENERATE_TEST_HERE
        tl.store(out_ptr.to(tl.pointer_type(val.dtype)), val)

    kernel_patched = patch_kernel(kernel, {'GENERATE_TEST_HERE': f"{literal}"})
    out = torch.zeros((1, ), dtype=torch.float32, device=device)
    h = kernel_patched[(1, )](out)
    assert re.search(r"arith.constant .* : " + dtype_str, h.asm["ttir"]) is not None


@triton.jit
def pass_const(a, b, choose_b):
    if choose_b:
        return b
    else:
        return a


@pytest.mark.parametrize("choose_const", [True, False])
@pytest.mark.parametrize("constexpr", [True, False])
@pytest.mark.parametrize("mode", ["direct", "call", "ternary", "if"])
def test_const(device, choose_const, constexpr, mode):

    @triton.jit(do_not_specialize=["choose_const"])
    def kernel(in_ptr: tl.const, out, c_out: tl.const, choose_const, n_elems: tl.int32, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elems
        val = tl.load(in_ptr + offsets, mask=mask)
        LOSE_TAIL
        tl.store(final_out + offsets, val, mask=mask)

    @triton.jit
    def kernel_constexpr(in_ptr: tl.const, out, c_out: tl.const, choose_const: tl.constexpr, n_elems: tl.int32,
                         BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elems
        val = tl.load(in_ptr + offsets, mask=mask)
        LOSE_TAIL
        tl.store(final_out + offsets, val, mask=mask)

    if mode == "direct":
        if choose_const:
            LOSE_TAIL = "final_out = c_out"
        else:
            LOSE_TAIL = "final_out = out"
    elif mode == "call":
        LOSE_TAIL = "final_out = pass_const(out, c_out, choose_const)"
    elif mode == "ternary":
        LOSE_TAIL = "final_out = c_out if choose_const else out"
    elif mode == "if":
        LOSE_TAIL = """
    if choose_const:
        final_out = c_out
    else:
        final_out = out
"""

    SIZE = 128
    input = torch.randn((SIZE, ), dtype=torch.float32, device=device)
    output = torch.zeros((SIZE, ), dtype=torch.float32, device=device)
    patched_kernel = patch_kernel(kernel_constexpr if constexpr else kernel, {'LOSE_TAIL': LOSE_TAIL, 'CONSTEXPR': ''})

    expect_fail = (not constexpr and mode != "direct") or choose_const
    if expect_fail:
        with pytest.raises(triton.CompilationError) as exc_info:
            patched_kernel[(1, )](input, output, output, choose_const, SIZE, SIZE)
        if constexpr:
            error = "Cannot store to a constant pointer"
        else:
            if mode == "call":
                error = "Inconsistent return types"
            elif mode == "if":
                error = "Mismatched type for final_out"
            elif mode == "ternary":
                error = "Ternary expression with dynamic condition has inconsistent type"
            else:
                assert mode == "direct" and choose_const
                error = "Cannot store to a constant pointer"
        error_msg = exc_info.value.error_message or str(exc_info.value.__cause__)
        assert error in error_msg, "Wrong error message!"
    else:
        patched_kernel[(1, )](input, output, output, choose_const, SIZE, SIZE)
        assert torch.all(input == output)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ['float32', 'float16'])
def test_dot_without_load(dtype_str, device):

    @triton.jit
    def _kernel(out):
        a = GENERATE_TEST_HERE
        b = GENERATE_TEST_HERE
        c = tl.dot(a, b)
        out_ptr = out + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
        tl.store(out_ptr, c)

    kernel = patch_kernel(_kernel, {'GENERATE_TEST_HERE': f"tl.full((32, 32), 1.0, tl.{dtype_str})"})
    a = torch.ones((32, 32), dtype=getattr(torch, dtype_str), device=device)
    b = torch.ones((32, 32), dtype=getattr(torch, dtype_str), device=device)
    out_ref = torch.matmul(a, b)
    out = torch.zeros((32, 32), dtype=getattr(torch, dtype_str), device=device)
    kernel[(1, )](out)
    assert torch.all(out == out_ref)


# ---------------
# test arange
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("start", [0, 1, 7, 16])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_arange(start, num_ctas, device):
    BLOCK = 128
    z_tri = torch.empty(BLOCK, dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(z, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
        off = tl.arange(0, BLOCK)
        val = tl.arange(START, END)
        tl.store(z + off, val)

    _kernel[(1, )](z_tri, START=start, END=start + BLOCK, BLOCK=BLOCK, num_ctas=num_ctas)
    z_ref = torch.arange(start, BLOCK + start, dtype=torch.int32, device=device)
    np.testing.assert_allclose(to_numpy(z_tri), to_numpy(z_ref))


# ---------------
# test load
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str, size, size_diff, other", [(dtype_str, size, size_diff, other)
                                                               for dtype_str in torch_dtypes
                                                               for size in [128, 512]
                                                               for size_diff in [0, 1, 2, 3, 4]
                                                               for other in [0, 1]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_masked_load(dtype_str, size, size_diff, other, num_ctas, device):
    dtype = getattr(torch, dtype_str)
    check_type_supported(dtype, device)  # bfloat16 on cc < 80 will not be tested

    input_size = size - size_diff
    output_size = size
    if dtype_str == 'bool':
        input = torch.randint(0, 2, (input_size, ), dtype=dtype, device=device)
    elif dtype_str in int_dtypes or dtype_str in uint_dtypes:
        input = torch.randint(0, 127, (input_size, ), dtype=dtype, device=device)
    else:
        input = torch.rand(input_size, dtype=dtype, device=device)
    output = torch.zeros((output_size, ), dtype=dtype, device=device)

    @triton.jit
    def _kernel(in_ptr, out_ptr, in_size: tl.constexpr, out_size: tl.constexpr):
        in_offsets = tl.arange(0, out_size)
        # Load inputs.
        x = GENERATE_TEST_HERE
        # Store output
        output_offsets = tl.arange(0, out_size)
        tl.store(out_ptr + output_offsets, x)

    mask_str = f"mask=in_offsets < in_size, other={other}" if size_diff > 0 else "None"
    kernel = patch_kernel(_kernel, {'GENERATE_TEST_HERE': f"tl.load(in_ptr + in_offsets, {mask_str})"})
    kernel[(1, )](input, output, input_size, output_size, num_ctas=num_ctas)

    reference_out = torch.cat((input, torch.full((size_diff, ), other, dtype=dtype, device=device)))
    torch.testing.assert_close(output, reference_out)


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
@pytest.mark.parametrize("mask_val", [True, False])
@pytest.mark.parametrize("other_val", [0, 1])
def test_masked_load_scalar(num_ctas, mask_val, other_val, device):
    input_val = 4.0
    size = 128
    dtype = torch.float32
    input = torch.full((size, ), input_val, dtype=dtype, device=device)
    output = torch.zeros((size, ), dtype=dtype, device=device)

    @triton.jit
    def kernel(in_ptr, out_ptr, size: tl.constexpr, mask: tl.constexpr, other: tl.constexpr):
        offsets = tl.arange(0, size)
        x = tl.load(in_ptr + offsets, mask=mask, other=other)
        tl.store(out_ptr + offsets, x)

    kernel[(1, )](input, output, size, mask_val, other_val, num_ctas=num_ctas)

    if mask_val:
        reference_out = torch.full((size, ), input_val, dtype=dtype, device=device)
    else:
        reference_out = torch.full((size, ), other_val, dtype=dtype, device=device)

    torch.testing.assert_close(output, reference_out)


# Testing masked loads with a copy to shared memory.
# FIXME: Shape too small for ldmatrix when num_ctas=4
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_masked_load_shared_memory(dtype, device):

    check_type_supported(dtype, device)  # bfloat16 on cc < 80 will not be tested

    M = 32
    N = 32
    K = 16

    in1 = torch.rand((M, K), dtype=dtype, device=device)
    in2 = torch.rand((K, N), dtype=dtype, device=device)
    out = torch.zeros((M, N), dtype=dtype, device=device)

    @triton.jit
    def _kernel(in1_ptr, in2_ptr, output_ptr, in_stride, in2_stride, out_stride, in_numel, in2_numel, out_numel,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):

        M_offsets = tl.arange(0, M)
        N_offsets = tl.arange(0, N)
        K_offsets = tl.arange(0, K)

        in_offsets = M_offsets[:, None] * in_stride + K_offsets[None, :]
        in2_offsets = K_offsets[:, None] * in2_stride + N_offsets[None, :]

        # Load inputs.
        x = tl.load(in1_ptr + in_offsets, mask=in_offsets < M * K)
        w = tl.load(in2_ptr + in2_offsets, mask=in2_offsets < K * N)

        # Without a dot product the memory doesn't get promoted to shared.
        o = tl.dot(x, w, out_dtype=tl.float32)

        # Store output
        output_offsets = M_offsets[:, None] * out_stride + N_offsets[None, :]
        tl.store(output_ptr + output_offsets, o, mask=output_offsets < M * N)

    pgm = _kernel[(1, )](in1, in2, out, in1.stride()[0], in2.stride()[0], out.stride()[0], in1.numel(), in2.numel(),
                         out.numel(), M=M, N=N, K=K)

    reference_out = torch.matmul(in1, in2)
    torch.testing.assert_close(out, reference_out, atol=1e-2, rtol=0)


@pytest.mark.interpreter
@pytest.mark.parametrize("cache", ["", ".ca", ".cg", ".cv"])
def test_load_cache_modifier(cache, device):
    src = torch.empty(128, device=device)
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst, src, CACHE: tl.constexpr):
        offsets = tl.arange(0, 128)
        x = tl.load(src + offsets, cache_modifier=CACHE)
        tl.store(dst + offsets, x)

    pgm = _kernel[(1, )](dst, src, CACHE=cache)

    if is_hip():
        target_arch = get_arch()
        # TODO: support testing for remaining architectures
        if 'gfx94' not in target_arch:
            return
        amdgcn = pgm.asm['amdgcn']
        cg_cache_modifier_str = 'nt'
        cv_cache_modifier_str = 'sc0 sc1'
        buffer_load_line = [line for line in amdgcn.splitlines() if "buffer_load" in line]
        global_load_line = [line for line in amdgcn.splitlines() if "global_load" in line]
        load_line = global_load_line[0] if global_load_line else buffer_load_line[0]
        if cache == '' or cache == '.ca':
            assert cg_cache_modifier_str not in load_line
        if cache == '.cg':
            assert cg_cache_modifier_str in load_line
        if cache == '.cv':
            assert cv_cache_modifier_str in load_line

    if is_cuda():
        ptx = pgm.asm['ptx']
        if cache == '':
            assert 'ld.global.ca' not in ptx
            assert 'ld.global.cg' not in ptx
        if cache == '.cg':
            assert 'ld.global.cg' in ptx
            assert 'ld.global.ca' not in ptx
        if cache == '.ca':
            assert 'ld.global.ca' in ptx
            assert 'ld.global.cg' not in ptx


@pytest.mark.interpreter
@pytest.mark.parametrize("N", [16, 10, 11, 1024])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_vectorization(N, num_ctas, device):
    block_size = 1024 * num_ctas
    src = torch.randn(block_size, device=device)
    dst = torch.empty(block_size, device=device)

    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    pgm = _kernel[(1, )](dst, src, N=N, BLOCK_SIZE=block_size)

    if not is_cuda():
        return

    ptx = pgm.asm["ptx"]
    if N % 16 == 0:
        assert "ld.global.v4.b32" in ptx
    else:
        assert "ld.global.b32" in ptx
    torch.testing.assert_close(dst[:N], src[:N], atol=1e-6, rtol=0)


@pytest.mark.interpreter
@pytest.mark.parametrize("has_hints", [False, True])
def test_vectorization_hints(has_hints, device):
    src = torch.empty(1024, device=device)
    dst = torch.empty(1024, device=device)
    off = torch.zeros(1, device=device, dtype=torch.int32)

    @triton.jit
    def _kernel(dst, src, off, N, BLOCK_SIZE: tl.constexpr, HINT: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offsets = offsets + tl.load(off)
        if HINT:
            tl.max_contiguous(tl.multiple_of(offsets, 1024), 1024)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    pgm = _kernel[(1, )](dst, src, off, N=1024, BLOCK_SIZE=src.shape[0], HINT=has_hints)
    if not is_cuda():
        return

    ptx = pgm.asm["ptx"]
    if has_hints:
        assert "ld.global.v4.b32" in ptx
    else:
        assert "ld.global.v4.b32" not in ptx


@pytest.mark.interpreter
def test_assume(device):

    @triton.jit
    def _kernel(out_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
        current_size = N - tl.program_id(0) * BLOCK_N
        tl.assume(current_size >= BLOCK_N)
        if current_size >= 128:
            tl.store(out_ptr + tl.program_id(0), current_size)
        else:
            tl.store(out_ptr + tl.program_id(0), current_size + 101024)

    output = torch.zeros(1024 // 128, device=device)
    pgm = _kernel[(1024 // 128, )](output, N=1024, BLOCK_N=128)

    if is_interpreter():
        return
