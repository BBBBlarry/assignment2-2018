from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: A(*i) + const_k)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: A(*i) * const_k)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""

    ZERO = tvm.const(0, dtype)

    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: tvm.max(A(*i), ZERO))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    ZERO = tvm.const(0, dtype)

    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="A_grad")
    C = tvm.compute(A.shape, lambda *i: tvm.select((A(*i) > ZERO), B(*i), ZERO))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""

    #raise Exception("NOT DONE YET!!!")
    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")

    if not transposeA and not transposeB:
        k = tvm.reduce_axis((0, shapeA[1]), name='k')
        C = tvm.compute((shapeA[0], shapeB[1]),
            lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k))
    elif not transposeA and transposeB:
        k = tvm.reduce_axis((0, shapeA[1]), name='k')
        C = tvm.compute((shapeA[0], shapeB[0]),
            lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k))
    elif transposeA and not transposeB:
        k = tvm.reduce_axis((0, shapeA[0]), name='k')
        C = tvm.compute((shapeA[1], shapeB[1]),
            lambda i, j: tvm.sum(A[k, i] * B[k, j], axis=k))
    else: # transposeA and transposeB
        k = tvm.reduce_axis((0, shapeA[0]), name='k')
        C = tvm.compute((shapeA[1], shapeB[0]),
            lambda i, j: tvm.sum(A[k, i] * B[j, k], axis=k))

    s = tvm.create_schedule(C.op)

    # optimizations
    BLOCK_SIZE = 100
    RED_AXIS_SPLIT = 4
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], BLOCK_SIZE, BLOCK_SIZE)
    ko, ki = s[C].split(k, factor=RED_AXIS_SPLIT)
    # reorder access pattern to improve A's access pattern
    s[C].reorder(xo, yo, ko, xi, ki, yi)
    # uniform access, so vectorize
    s[C].vectorize(yi)
    # multithreading on blocks
    s[C].parallel(xo)

    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)

    '''
    # evalulate time
    ctx = tvm.context(tgt, 0)
    # data
    shapeX = (500, 700)
    shapeY = (700, 1000)
    shapeZ = (500, 1000)
    a = tvm.nd.array(np.random.rand(*shapeX).astype(dtype), ctx)
    b = tvm.nd.array(np.random.rand(*shapeY).astype(dtype), ctx)
    z = tvm.nd.array(np.zeros(shapeZ).astype(dtype), ctx)

    # do it
    evaluator = f.time_evaluator(f.entry_name, ctx, number=10)
    print('Time: %f' % evaluator(a,b,z).mean)

    raise Exception()
    '''

    return f


def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""

    # stride = 1; padding = 0
    shapeZ = (N, M, H - R + 1, W - S + 1)

    X = tvm.placeholder(shapeX, dtype = dtype, name = "X")
    F = tvm.placeholder(shapeF, dtype = dtype, name = "F")

    rr = tvm.reduce_axis((0, R), name = "rh")
    rs = tvm.reduce_axis((0, S), name = "rs")
    rc = tvm.reduce_axis((0, C), name = "rc")

    Z = tvm.compute(shapeZ, lambda n, m, i, j: tvm.sum(X[n, rc, i + rr, j + rs] * F[m, rc, rr, rs], axis = [rc, rr, rs]), name="Z")
    s = tvm.create_schedule(Z.op)

    #print(tvm.lower(s, [X, F, Z], simple_mode=True))
    f = tvm.build(s, [X, F, Z], tgt, target_host=tgt_host, name=func_name)

    '''
    # evalulate time
    ctx = tvm.context(tgt, 0)
    # data
    shapeX = (100, 3, 28, 28)
    shapeF = (10, 3, 5, 5)
    shapeY = (100, 10, 24, 24)

    a = np.random.uniform(0, 10, size=shapeX).astype(dtype)
    b = np.random.uniform(0, 10, size=shapeF).astype(dtype)
    c = np.zeros(shapeY).astype(dtype)
    arr_a = tvm.nd.array(a, ctx=ctx)
    arr_b = tvm.nd.array(b, ctx=ctx)
    arr_c = tvm.nd.array(c, ctx=ctx)

    # do it
    evaluator = f.time_evaluator(f.entry_name, ctx, number=100)
    print('Time: %f' % evaluator(arr_a,arr_b,arr_c).mean)
    raise Exception()
    '''

    return f



def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    raise Exception("NOT FINISHED")
    X = tvm.placeholder(shape, dtype = dtype, name = "X")

    rj = tvm.reduce_axis((0, shape[1]), name = "rj")
    rej = tvm.reduce_axis((0, shape[1]), name = "rej")

    m_X = tvm.compute((shape[0],), lambda i: tvm.max(X[i, rj], axis = rj), name = "m_X")
    e_X = tvm.compute(shape, lambda i, j: tvm.exp(X[i, j] - m_X[i]), name = "e_X")
    s_eX = tvm.compute((shape[0],), lambda i: tvm.sum(e_X[i, rej], axis = rej), name = "s_eX")
    Z = tvm.compute(shape, lambda i, j: e_X[i, j] / s_eX[i], name = "Z")

    s = tvm.create_schedule([Z.op])

    print(tvm.lower(s, [X], simple_mode=True))
    raise Exceptions()

    f = tvm.build(s, [X], tgt, target_host=tgt_host, name=func_name)

    return f


def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f
