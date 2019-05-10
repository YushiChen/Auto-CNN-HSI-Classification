from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x1',
    'avg_pool_3x1',
    'skip_connect',
    'conv_3x1',
    'conv_5x1',
    'conv_7x1',
    'conv_9x1',
]


HSI = Genotype(normal=[('conv_7x1', 0), ('avg_pool_3x1', 1), ('conv_3x1', 0), ('conv_7x1', 1), ('conv_5x1', 1), ('conv_5x1', 3), ('conv_7x1', 3), ('conv_5x1', 2)], normal_concat=range(2, 6), reduce=[('conv_7x1', 0), ('avg_pool_3x1', 1), ('conv_7x1', 0), ('avg_pool_3x1', 2), ('conv_7x1', 0), ('avg_pool_3x1', 1), ('avg_pool_3x1', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))

