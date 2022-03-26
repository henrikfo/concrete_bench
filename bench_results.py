import codecs
import json
import math
import numpy as np
import sys
import os
import csv
import timeit

sys.path.append(".")

path = "target/criterion"
types = ["LWE", "VectorLWE_10", "teewondee"]
operations = ["Encrypt", "Decrypt", "Addition", "Real Addition", "Sum ciphertext vector", "Interger Multiplication", \
              "Real Multiplication", "Ciphertext mult", "Bootstrap_w_func", "Keyswitch", "Encrypt vector", \
              "Decrypt vector", "Dot prod", "Real dot prod"]
lwe_keys = ["512", "1024", "2048", "4096"]

def get_time_str(t, ):
    times = ["ns", "\u03BCs", "ms", "s"]
    ind = math.floor((math.log10(t))/3)
    
    d = math.floor(math.log10(t))
    d2 = math.floor((math.log10(t))%3+1)
    t = round(t/10**(d-2))
    t = t/10**(3-d2)
    t = ('%f' % t).rstrip('0').rstrip('.') if d2 == 3 else str(t)
    return t+' '+times[ind]

Rust_Bench = {}
for typ in types:
    op_dict = {}
    for op in operations:
        key_dict = {}
        for key in lwe_keys:
            file = path+'/'+typ+'/'+op+'/'+key
            try:
                f = open(file+'/new/estimates.json')
                data = json.load(f)
                mean = data['mean']['point_estimate']
                key_dict[key] = get_time_str(mean)
                f.close()
            except:
                continue
        op_dict[op] = key_dict
    Rust_Bench[typ] = op_dict

plain = {'Scalar': {}}

def add_single(x):
    return x+x

def mult_int(x):
    return x*0

def mult_real(x):
    return x*0.00

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def add_vector(x):
    return x+x

def add_vector2(x):
    y = x+x
    return y

def mult_vector(x, p):
    return x*p

def dot_vector(x):
    return np.dot(x, x)

def sum_vector(x):
    return sum(x)

def in_nano(x):
    return float("{:.5f}".format(x*1e9))
    

num_runs = 1000
plain['Scalar']['Addition'] = in_nano(timeit.Timer(lambda: add_single(0)).timeit(number = num_runs)/num_runs)
#print(timeit.Timer(lambda: add_single(0)).timeit(number = num_runs)/num_runs))

plain['Scalar']['Int Multiplication'] = in_nano(timeit.Timer(lambda: mult_int(0)).timeit(number = num_runs)/num_runs)

plain['Scalar']['Real Multiplication'] = in_nano(timeit.Timer(lambda: mult_real(0)).timeit(number = num_runs)/num_runs)

plain['Scalar']['f'] = in_nano(timeit.Timer(lambda: sigmoid(0.)).timeit(number = num_runs)/num_runs)

csv_file = "bench.csv"
csv_columns = ['plain', lwe_keys[0], lwe_keys[1], lwe_keys[2], lwe_keys[3]]

with open('bench.json', 'w') as outfile:
    json.dump(Rust_Bench, outfile)