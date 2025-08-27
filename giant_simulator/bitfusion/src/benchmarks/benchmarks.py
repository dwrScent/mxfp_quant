import argparse
import logging

from dnnweaver2.graph import Graph, get_default_graph
from dnnweaver2.tensorOps.cnn import conv2D, maxPool, flatten, matmul, addBias, batch_norm, reorg, concat, leakyReLU, add
from dnnweaver2 import get_tensor
import logging
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint

import bitfusion.src.benchmarks.ant_bench as ant

import bitfusion.src.benchmarks.bitfusion_bench as bit
import bitfusion.src.benchmarks.tender_bench as tender


# import bitfusion.src.benchmarks.ant_weight_bench as ant_weight
# import bitfusion.src.benchmarks.biscaled_bench as biscaled

import bitfusion.src.benchmarks.olive_bench as olive

import bitfusion.src.benchmarks.bitfusion_bench as bf
import bitfusion.src.benchmarks.giant_bench as giant

import bitfusion.src.benchmarks.microscopiq_bench as microscopiq
import bitfusion.src.benchmarks.m2xfp_bench as m2xfp

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

def create_net(net_name, net_list, batch_size, asymmetry=False, mode='default'):
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
                    out = create_fc(input_size, kernel_size, c_dtype=precision, w_dtype=precision, asymmetry=asymmetry, mode=mode)
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

def create_fc(input_size, weight_size, c_dtype=None, w_dtype=None, asymmetry=False, mode='default'):
    batch_size = input_size[0]
    output_channels = weight_size[0]
    input_channels = weight_size[1]

    # add by wmhu. codeant w4a8
    if mode == 'awq':
        input_dtype = FQDtype.FXP16
        # w_dtype = FQDtype.FXP16
        
        # input_dtype = w_dtype

    elif mode == 'giant':
        input_dtype = FQDtype.FXP8
    else:
        input_dtype = w_dtype
    # if asymmetry:
    #     input_dtype = FQDtype.FXP8
    # else:
    #     input_dtype = w_dtype
    input = get_tensor(shape=(batch_size, input_size[1]), name='data', dtype=input_dtype, trainable=False)
    weights = get_tensor(shape=(output_channels, input_channels), name='weights', dtype=w_dtype)
    biases = get_tensor(shape=(output_channels,), name='biases', dtype=c_dtype)
    _fc = matmul(input, weights, biases, dtype=c_dtype)
    return _fc

benchlist = [\
            #  'vgg16',
            #  'resnet18',
            #  'resnet50',
            #  'inceptionv3',
            #  'vit', 
            #  'mnli',
            #  'cola', 
            #  'sst_2',
    
            # 'bart_base',
            # 'bert_base',
            # 'bert_large',
            # 'gpt2_xl',
            # 'bloom3b',
            # 'bloom7b1',
            # 'opt_13b',
            # 'llama_7b',
            # 'llama_13b',
            # 'llama_30b',
            # 'llama_65b',
            # 'llama2_13b',
            'llama2_7b',
            'opt6b7',
            'falcon_7b',
            'llama3_8b',
            'mistral_7b',
            'llama3_70b',            
            ]


def get_bench_nn_ant(bench_name, batch_size):
    if bench_name == 'vgg16':
        return create_net(bench_name, ant.vgg16, batch_size)
    elif bench_name == 'resnet18':
        return create_net(bench_name, ant.resnet18, batch_size)
    elif bench_name == 'resnet50':
        return create_net(bench_name, ant.resnet50, batch_size)
    elif bench_name == 'inceptionv3':
        return create_net(bench_name, ant.inceptionv3, batch_size)
    elif bench_name == 'vit':
        return create_net(bench_name, ant.vit, batch_size)
    elif bench_name == 'rte':
        return create_net(bench_name, ant.rte, batch_size)
    elif bench_name == 'wnli':
        return create_net(bench_name, ant.wnli, batch_size)
    elif bench_name == 'mrpc':
        return create_net(bench_name, ant.mrpc, batch_size)
    elif bench_name == 'cola':
        return create_net(bench_name, ant.cola, batch_size)
    elif bench_name == 'sst_2':
        return create_net(bench_name, ant.sst_2, batch_size)
    elif bench_name == 'qnli':
        return create_net(bench_name, ant.qnli, batch_size)
    elif bench_name == 'qqp':
        return create_net(bench_name, ant.qqp, batch_size)
    elif bench_name == 'mnli':
        return create_net(bench_name, ant.mnli, batch_size)
    elif bench_name == 'bart_base':
        return create_net(bench_name, ant.bart_base, batch_size)
    elif bench_name == 'bert_base':
        return create_net(bench_name, ant.bert_base, batch_size)
    elif bench_name == 'bert_large':
        return create_net(bench_name, ant.bert_large, batch_size)
    elif bench_name == 'gpt2_xl':
        return create_net(bench_name, ant.gpt2_xl, batch_size)
    elif bench_name == 'bloom3b':
        return create_net(bench_name, ant.bloom3b, batch_size)
    elif bench_name == 'bloom7b1':
        return create_net(bench_name, ant.bloom7b1, batch_size)
    elif bench_name == 'opt6b7':
        return create_net(bench_name, ant.opt6b7, batch_size)
    elif bench_name == 'opt_13b':
        return create_net(bench_name, ant.opt_13b, batch_size)
    elif bench_name == 'llama_7b':
        return create_net(bench_name, ant.llama_7b, batch_size)
    elif bench_name == 'llama_13b':
        return create_net(bench_name, ant.llama_13b, batch_size)
    elif bench_name == 'llama_30b':
        return create_net(bench_name, ant.llama_30b, batch_size)
    elif bench_name == 'llama_65b':
        return create_net(bench_name, ant.llama_65b, batch_size)
    elif bench_name == 'llama2_7b':
        return create_net(bench_name, ant.llama2_7b, batch_size)
    elif bench_name == 'llama2_13b':
        return create_net(bench_name, ant.llama2_13b, batch_size)
    elif bench_name == 'falcon_7b':
        return create_net(bench_name, ant.falcon_7b, batch_size)
    elif bench_name == 'llama3_8b':
        return create_net(bench_name, ant.llama3_8b, batch_size)
    elif bench_name == 'llama3_70b':
        return create_net(bench_name, ant.llama3_70b, batch_size)
    elif bench_name == 'mistral_7b':
        return create_net(bench_name, ant.mistral_7b, batch_size)
    elif bench_name == 'opt6b7':
        return create_net(bench_name, ant.opt6b7, batch_size)
    
def get_bench_nn_olive(bench_name, batch_size):
    if bench_name == 'vgg16':
        return create_net(bench_name, olive.vgg16, batch_size)
    elif bench_name == 'resnet18':
        return create_net(bench_name, olive.resnet18, batch_size)
    elif bench_name == 'resnet50':
        return create_net(bench_name, olive.resnet50, batch_size)
    elif bench_name == 'inceptionv3':
        return create_net(bench_name, olive.inceptionv3, batch_size)
    elif bench_name == 'vit':
        return create_net(bench_name, olive.vit, batch_size)
    elif bench_name == 'rte':
        return create_net(bench_name, olive.rte, batch_size)
    elif bench_name == 'wnli':
        return create_net(bench_name, olive.wnli, batch_size)
    elif bench_name == 'mrpc':
        return create_net(bench_name, olive.mrpc, batch_size)
    elif bench_name == 'cola':
        return create_net(bench_name, olive.cola, batch_size)
    elif bench_name == 'sst_2':
        return create_net(bench_name, olive.sst_2, batch_size)
    elif bench_name == 'qnli':
        return create_net(bench_name, olive.qnli, batch_size)
    elif bench_name == 'qqp':
        return create_net(bench_name, olive.qqp, batch_size)
    elif bench_name == 'mnli':
        return create_net(bench_name, olive.mnli, batch_size)
    elif bench_name == 'bart_base':
        return create_net(bench_name, olive.bart_base, batch_size)
    elif bench_name == 'bert_base':
        return create_net(bench_name, olive.bert_base, batch_size)
    elif bench_name == 'bert_large':
        return create_net(bench_name, olive.bert_large, batch_size)
    elif bench_name == 'gpt2_xl':
        return create_net(bench_name, olive.gpt2_xl, batch_size)
    elif bench_name == 'bloom3b':
        return create_net(bench_name, olive.bloom3b, batch_size)
    elif bench_name == 'bloom7b1':
        return create_net(bench_name, olive.bloom7b1, batch_size)
    elif bench_name == 'opt6b7':
        return create_net(bench_name, olive.opt6b7, batch_size)
    elif bench_name == 'opt_13b':
        return create_net(bench_name, olive.opt_13b, batch_size)
    elif bench_name == 'llama_7b':
        return create_net(bench_name, olive.llama_7b, batch_size)
    elif bench_name == 'llama_13b':
        return create_net(bench_name, olive.llama_13b, batch_size)
    elif bench_name == 'llama_30b':
        return create_net(bench_name, olive.llama_30b, batch_size)
    elif bench_name == 'llama_65b':
        return create_net(bench_name, olive.llama_65b, batch_size)
    elif bench_name == 'llama2_7b':
        return create_net(bench_name, olive.llama2_7b, batch_size)
    elif bench_name == 'llama2_13b':
        return create_net(bench_name, olive.llama2_13b, batch_size)
    elif bench_name == 'falcon_7b':
        return create_net(bench_name, olive.falcon_7b, batch_size)
    elif bench_name == 'llama3_8b':
        return create_net(bench_name, olive.llama3_8b, batch_size)
    elif bench_name == 'llama3_70b':
        return create_net(bench_name, olive.llama3_70b, batch_size)
    elif bench_name == 'mistral_7b':
        return create_net(bench_name, olive.mistral_7b, batch_size)
    elif bench_name == 'opt6b7':
        return create_net(bench_name, olive.opt6b7, batch_size)
    
def get_bench_nn_bit(bench_name, batch_size):
    if bench_name == 'gpt2_xl':
        return create_net(bench_name, bit.gpt2_xl, batch_size)
    elif bench_name == 'bloom3b':
        return create_net(bench_name, bit.bloom3b, batch_size)
    elif bench_name == 'bloom7b1':
        return create_net(bench_name, bit.bloom7b1, batch_size)
    elif bench_name == 'opt6b7':
        return create_net(bench_name, bit.opt6b7, batch_size)
    elif bench_name == 'opt_13b':
        return create_net(bench_name, bit.opt_13b, batch_size)
    elif bench_name == 'llama_7b':
        return create_net(bench_name, bit.llama_7b, batch_size)
    elif bench_name == 'llama_13b':
        return create_net(bench_name, bit.llama_13b, batch_size)
    elif bench_name == 'llama_30b':
        return create_net(bench_name, bit.llama_30b, batch_size)
    elif bench_name == 'llama_65b':
        return create_net(bench_name, bit.llama_65b, batch_size)
    elif bench_name == 'llama2_7b':
        return create_net(bench_name, bit.llama2_7b, batch_size)
    elif bench_name == 'llama2_13b':
        return create_net(bench_name, bit.llama2_13b, batch_size)
    elif bench_name == 'bert_base':
        return create_net(bench_name, bit.bert_base, batch_size)
    
def get_bench_nn_tender(bench_name, batch_size):
    if bench_name == 'gpt2_xl':
        return create_net(bench_name, tender.gpt2_xl, batch_size)
    elif bench_name == 'bloom3b':
        return create_net(bench_name, tender.bloom3b, batch_size)
    elif bench_name == 'bloom7b1':
        return create_net(bench_name, tender.bloom7b1, batch_size)
    elif bench_name == 'opt6b7':
        return create_net(bench_name, tender.opt6b7, batch_size)
    elif bench_name == 'opt_13b':
        return create_net(bench_name, tender.opt_13b, batch_size)
    elif bench_name == 'llama_7b':
        return create_net(bench_name, tender.llama_7b, batch_size)
    elif bench_name == 'llama_13b':
        return create_net(bench_name, tender.llama_13b, batch_size)
    elif bench_name == 'llama_30b':
        return create_net(bench_name, tender.llama_30b, batch_size)
    elif bench_name == 'llama_65b':
        return create_net(bench_name, tender.llama_65b, batch_size)
    elif bench_name == 'llama2_7b':
        return create_net(bench_name, tender.llama2_7b, batch_size)
    elif bench_name == 'llama2_13b':
        return create_net(bench_name, tender.llama2_13b, batch_size)
    elif bench_name == 'bert_base':
        return create_net(bench_name, tender.bert_base, batch_size)

def get_bench_nn_m2xfp(bench_name, batch_size):
    if bench_name == 'gpt2_xl':
        return create_net(bench_name, m2xfp.gpt2_xl, batch_size)
    elif bench_name == 'bloom3b':
        return create_net(bench_name, m2xfp.bloom3b, batch_size)
    elif bench_name == 'bloom7b1':
        return create_net(bench_name, m2xfp.bloom7b1, batch_size)
    elif bench_name == 'opt6b7':
        return create_net(bench_name, m2xfp.opt6b7, batch_size)
    elif bench_name == 'opt_13b':
        return create_net(bench_name, m2xfp.opt_13b, batch_size)
    elif bench_name == 'llama_7b':
        return create_net(bench_name, m2xfp.llama_7b, batch_size)
    elif bench_name == 'llama_13b':
        return create_net(bench_name, m2xfp.llama_13b, batch_size)
    elif bench_name == 'llama_30b':
        return create_net(bench_name, m2xfp.llama_30b, batch_size)
    elif bench_name == 'llama_65b':
        return create_net(bench_name, m2xfp.llama_65b, batch_size)
    elif bench_name == 'llama2_7b':
        return create_net(bench_name, m2xfp.llama2_7b, batch_size)
    elif bench_name == 'llama2_13b':
        return create_net(bench_name, m2xfp.llama2_13b, batch_size)
    elif bench_name == 'bert_base':
        return create_net(bench_name, m2xfp.bert_base, batch_size)
    elif bench_name == 'falcon_7b':
        return create_net(bench_name, m2xfp.falcon_7b, batch_size)
    elif bench_name == 'llama3_8b':
        return create_net(bench_name, m2xfp.llama3_8b, batch_size)
    elif bench_name == 'llama3_70b':
        return create_net(bench_name, m2xfp.llama3_70b, batch_size)
    elif bench_name == 'mistral_7b':
        return create_net(bench_name, m2xfp.mistral_7b, batch_size)
    elif bench_name == 'opt6b7':
        return create_net(bench_name, m2xfp.opt6b7, batch_size)

def get_bench_nn_microscopiq(bench_name, batch_size):
    if bench_name == 'gpt2_xl':
        return create_net(bench_name, microscopiq.gpt2_xl, batch_size)
    elif bench_name == 'bloom3b':
        return create_net(bench_name, microscopiq.bloom3b, batch_size)
    elif bench_name == 'bloom7b1':
        return create_net(bench_name, microscopiq.bloom7b1, batch_size)
    elif bench_name == 'opt6b7':
        return create_net(bench_name, microscopiq.opt6b7, batch_size)
    elif bench_name == 'opt_13b':
        return create_net(bench_name, microscopiq.opt_13b, batch_size)
    elif bench_name == 'llama_7b':
        return create_net(bench_name, microscopiq.llama_7b, batch_size)
    elif bench_name == 'llama_13b':
        return create_net(bench_name, microscopiq.llama_13b, batch_size)
    elif bench_name == 'llama_30b':
        return create_net(bench_name, microscopiq.llama_30b, batch_size)
    elif bench_name == 'llama_65b':
        return create_net(bench_name, microscopiq.llama_65b, batch_size)
    elif bench_name == 'llama2_7b':
        return create_net(bench_name, microscopiq.llama2_7b, batch_size)
    elif bench_name == 'llama2_13b':
        return create_net(bench_name, microscopiq.llama2_13b, batch_size)
    elif bench_name == 'bert_base':
        return create_net(bench_name, microscopiq.bert_base, batch_size)
    elif bench_name == 'falcon_7b':
        return create_net(bench_name, microscopiq.falcon_7b, batch_size)
    elif bench_name == 'llama3_8b':
        return create_net(bench_name, microscopiq.llama3_8b, batch_size)
    elif bench_name == 'llama3_70b':
        return create_net(bench_name, microscopiq.llama3_70b, batch_size)
    elif bench_name == 'mistral_7b':
        return create_net(bench_name, microscopiq.mistral_7b, batch_size)
    elif bench_name == 'opt6b7':
        return create_net(bench_name, microscopiq.opt6b7, batch_size)
    
def get_bench_nn_giant(bench_name, batch_size):
    if bench_name == 'gpt2_xl':
        return create_net(bench_name, giant.gpt2_xl, batch_size)
    elif bench_name == 'bloom3b':
        return create_net(bench_name, giant.bloom3b, batch_size)
    elif bench_name == 'bloom7b1':
        return create_net(bench_name, giant.bloom7b1, batch_size)
    elif bench_name == 'opt6b7':
        return create_net(bench_name, giant.opt6b7, batch_size)
    elif bench_name == 'opt_13b':
        return create_net(bench_name, giant.opt_13b, batch_size)
    elif bench_name == 'llama_7b':
        return create_net(bench_name, giant.llama_7b, batch_size)
    elif bench_name == 'llama_13b':
        return create_net(bench_name, giant.llama_13b, batch_size)
    elif bench_name == 'llama_30b':
        return create_net(bench_name, giant.llama_30b, batch_size)
    elif bench_name == 'llama_65b':
        return create_net(bench_name, giant.llama_65b, batch_size)
    elif bench_name == 'llama2_7b':
        return create_net(bench_name, giant.llama2_7b, batch_size)
    elif bench_name == 'llama2_13b':
        return create_net(bench_name, giant.llama2_13b, batch_size)
    elif bench_name == 'bert_base':
        return create_net(bench_name, giant.bert_base, batch_size)
    elif bench_name == 'falcon_7b':
        return create_net(bench_name, giant.falcon_7b, batch_size)
    elif bench_name == 'llama3_8b':
        return create_net(bench_name, giant.llama3_8b, batch_size)
    elif bench_name == 'llama3_70b':
        return create_net(bench_name, giant.llama3_70b, batch_size)
    elif bench_name == 'mistral_7b':
        return create_net(bench_name, giant.mistral_7b, batch_size)
    elif bench_name == 'opt6b7':
        return create_net(bench_name, giant.opt6b7, batch_size)


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
