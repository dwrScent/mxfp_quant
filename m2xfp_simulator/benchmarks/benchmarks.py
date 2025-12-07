import argparse
import logging

from accelerator.src.graph import Graph, get_default_graph
from accelerator.src.tensorOps.cnn import conv2D, maxPool, flatten, matmul, addBias, batch_norm, reorg, concat, leakyReLU, add
from accelerator.src import get_tensor
import logging
from accelerator.src.scalar.dtypes import FQDtype, FixedPoint
import benchmarks.accel_model_configs as accel_model_configs
from typing import Dict, Any, List

import os

def fc(tensor_in, output_channels=1024,
        f_dtype=None, w_dtype=None,
        act='linear'):
    input_channels = tensor_in.shape[-1]
    weights = get_tensor(shape=(output_channels, input_channels),
            name='weights',
            dtype=w_dtype)
    biases = get_tensor(shape=(output_channels,),
            name='biases',
            dtype=FixedPoint(32,w_dtype.frac_bits + tensor_in.dtype.frac_bits))
    _fc = matmul(tensor_in, weights, biases, dtype=f_dtype)

    if act == 'leakyReLU':
        with get_default_graph().name_scope(act):
            act = leakyReLU(_fc, dtype=_fc.dtype)
    elif act == 'linear':
        with get_default_graph().name_scope(act):
            act = _fc
    else:
        raise (ValueError, 'Unknown activation type {}'.format(act))

    return act

def conv(tensor_in, filters=32, stride=None, kernel_size=3, pad='SAME',
        c_dtype=None, w_dtype=None,
        act='linear'):

    if stride is None:
        stride = (1,1,1,1)

    input_channels = tensor_in.shape[-1]

    weights = get_tensor(shape=(filters, kernel_size, kernel_size, input_channels),
                         name='weights',
                         dtype=w_dtype)
    biases = get_tensor(shape=(filters),
                         name='biases',
                         dtype=FixedPoint(32,w_dtype.frac_bits + tensor_in.dtype.frac_bits))
    _conv = conv2D(tensor_in, weights, biases, stride=stride, pad=pad, dtype=c_dtype)

    if act == 'leakyReLU':
        with get_default_graph().name_scope(act):
            act = leakyReLU(_conv, dtype=_conv.dtype)
    elif act == 'linear':
        with get_default_graph().name_scope(act):
            act = _conv
    else:
        raise (ValueError, 'Unknown activation type {}'.format(act))

    return act


def get_precision(precision):
    if precision == 16:
        return FQDtype.FXP16
    if precision == 8:
        return FQDtype.FXP8
    if precision == 4:
        return FQDtype.FXP4
    if precision == 6:
        return FQDtype.FXP6

def create_net(net_name, net_list, batch_size, mode='default'):
    g = Graph(net_name, dataset='imagenet', log_level=logging.INFO)
    with g.as_default():
        for idx, op in enumerate(net_list):
            input_size, kernel_size, output_size, kernel_stride, padding, precision, op_type =  op
            input_size[0] = input_size[0] * batch_size
            output_size[0] = output_size[0] * batch_size
            precision = get_precision(precision)

            if op_type == 0:
                with g.name_scope('conv'+str(idx)):
                    out = create_conv(input_size, kernel_size, stride_size=kernel_stride, pad=padding, c_dtype=FQDtype.FXP16, w_dtype=precision)
                    # print(idx, op, out.shape)
                    assert out.shape[0] == output_size[0]
                    assert out.shape[1] == output_size[2]
                    assert out.shape[2] == output_size[3]
                    assert out.shape[3] == output_size[1]
            else:
                with g.name_scope('fc'+str(idx)):
                    out = create_fc(input_size, kernel_size, c_dtype=precision, w_dtype=precision, mode=mode)
                    # print(idx, op, out.shape)
                    assert out.shape[0] == output_size[0]
                    assert out.shape[1] == output_size[1]
    return g

def create_conv(input_size, weight_size, stride_size=None, pad=None, c_dtype=None, w_dtype=None):

    if stride_size is None:
        stride = (1,1,1,1)
    else:
        stride = (1,stride_size[0],stride_size[1],1)

    batch_size = input_size[0]
    output_channels = weight_size[0]
    input_channels = weight_size[1]
    kernel_size = (weight_size[2], weight_size[3])

    input = get_tensor(shape=(batch_size, input_size[2], input_size[3], input_size[1]), name='data', dtype=w_dtype, trainable=False)
    weights = get_tensor(shape=(output_channels, kernel_size[0], kernel_size[1], input_channels), name='weights', dtype=w_dtype)
    biases = get_tensor(shape=(output_channels), name='biases', dtype=c_dtype)
    _conv = conv2D(input, weights, biases, stride=stride, pad=pad, dtype=c_dtype)
    return _conv

def create_fc(input_size, weight_size, c_dtype=None, w_dtype=None, mode='default'):
    batch_size = input_size[0]
    output_channels = weight_size[0]
    input_channels = weight_size[1]

    # add by wmhu. codeant w4a8
    if mode == 'awq':
        input_dtype = FQDtype.FXP16
        # w_dtype = FQDtype.FXP16
        # input_dtype = w_dtype
    elif mode == 'mant':
        input_dtype = FQDtype.FXP8
    else:
        input_dtype = w_dtype

    input = get_tensor(shape=(batch_size, input_size[1]), name='data', dtype=input_dtype, trainable=False)
    weights = get_tensor(shape=(output_channels, input_channels), name='weights', dtype=w_dtype)
    biases = get_tensor(shape=(output_channels,), name='biases', dtype=c_dtype)
    _fc = matmul(input, weights, biases, dtype=c_dtype)
    return _fc

benchlist = []


def get_bench_nn(accelerator: str, bench_name: str, batch_size: int):
    """
    Factory function to create a network graph for a given benchmark.
    """
    if accelerator not in accel_model_configs.accelerators:
        raise ValueError(
            f"Unknown accelerator/accelerator '{accelerator}'. "
            f"Available accelerators: {list(accel_model_configs.accelerators.keys())}"
        )
    
    if bench_name not in accel_model_configs.MODELS:
        raise ValueError(
            f"Unknown benchmark/model '{bench_name}'. "
            f"Supported models: {accel_model_configs.MODELS}"
        )
    
    net_list = accel_model_configs.generate_config(
        model_key=bench_name,
        accelerator_key=accelerator,
        seq_len=2048,
        repeated_blocks=1,
    )

    return create_net(f"{accelerator}_{bench_name}", net_list, batch_size)

def write_to_csv(csv_name, fields, stats, graph, csv_path='./'):
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    for l in stats:
        print(l)
        print(stats[l]['total'])

    bench_csv_name = os.path.join(csv_path, csv_name)
    with open(bench_csv_name, 'w') as f:
        f.write(', '.join(fields+['\n']))
        for l in network:
            if isinstance(network[l], ConvLayer):
                f.write('{}, {}\n'.format(l, ', '.join(str(x) for x in stats[l]['total'])))

def get_bench_numbers(graph, sim_obj, batch_size=1, weight_stationary = False):
    stats = {}
    for opname, op in graph.op_registry.items():
        out = sim_obj.get_cycles(op, batch_size, weight_stationary = weight_stationary)
        if out is not None:
            s, l = out
            stats[opname] = s
    return stats

if __name__ == "__main__":
    # parser object
    argp = argparse.ArgumentParser()

    # parser arguments
    argp.add_argument("-c", "--config_file", dest='config_file', default='conf.ini', type=str)
    argp.add_argument("-v", "--verbose", dest='verbose', default=False, action='store_true')

    # parse
    args = argp.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    # Read config file
    logger.info('Creating benchmarks')

    sim_obj = Simulator(args.config_file, args.verbose)
    fields = ['Layer', 'Total Cycles', 'Memory Stall Cycles', \
              'Activation Reads', 'Weight Reads', 'Output Reads', \
              'DRAM Reads', 'Output Writes', 'DRAM Writes']
    csv_dir = 'csv'
    if not os.path.isdir(csv_dir):
        os.makedirs(csv_dir)

    for bench in benchlist:
        print(bench)
        nn = get_bench_nn(bench)
        print(nn)
        stats = get_bench_numbers(nn, sim_obj, weight_stationary = False)
        write_to_csv(os.path.join(csv_dir, bench+'.csv'), fields, stats, nn)
