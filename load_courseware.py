#! /usr/bin/env python

#import copy
import my_util
#import re
#import sys

def load_stat21x_trans(filename, label_type='int'):
  with open(filename) as f:
    for line in f:
      if "++++++++++" in line or "==========" in line:
        continue
      items = line.strip().split('\t')
      if label_type == 'int':
        yield (my_util.str_to_sparse(items[0]), int(items[1]))
      elif label_type == 'strings':
        yield (my_util.str_to_sparse(items[0]), items[1].split(' '))

def load_stat21x_slides(filename):
  with open(filename) as f:
    for line in f:
      items = line.strip().split('_')
      yield my_util.str_to_sparse(items[1])

def load_external_fea(fea_dir, l_id, fea_weight=10):
  fea = []
  with open(fea_dir + '/' + l_id) as f:
    for line in f:
      fea.append(map(lambda x: fea_weight*float(x), line.strip().split('\t')))
  return fea

def load_stat21x_tx(filename):
  with open(filename) as f:
    for line in f:
      items = line.strip().split('\t')
      yield (my_util.str_to_sparse(items[0]), items[1])

def load_stat21x_seg(seg):
  with open(seg) as f:
    return [x.strip() for x in f.readlines()]

def load_stop_list():
  with open('/usr/users/swli/program/nlp_util/stopList') as f:
    return [x.strip() for x in f.readlines()]

