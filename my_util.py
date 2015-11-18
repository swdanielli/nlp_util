#! /usr/bin/env python

import copy
#import json
import re
import string
import sys
#import unicodedata
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()

def print_footer(f_o, resource_type):
  if resource_type == 'trans':
    f_o.write('==========\n')

def print_header(f_o, resource_type):
  if resource_type == 'trans':
    f_o.write('++++++++++\n')
    f_o.write('==========\n')

def print_fea(f_o, page_idx, fea, resource_type):
  if resource_type == 'slides':
    f_o.write(str(page_idx)
      + '_'
      + ' '.join([key + ':' + str(value) for key, value in fea.iteritems()])
      + '\n'
    )
  elif resource_type == 'trans':
    f_o.write(
      ' '.join([key + ':' + str(value) for key, value in fea.iteritems()])
      + '\t'
      + str(page_idx)
      + '\n'
    )
    f_o.write('++++++++++\n')

def load_dictionary_idf(filename):
  dictionary = load_dictionary(filename)
  with open(filename) as f:
    idf = {x.strip().split('\t')[1]: x.strip().split('\t')[2] for x in f.readlines()}
  return (dictionary, idf)

def load_dictionary(filename):
  with open(filename) as f:
    return {x.strip().split('\t')[0]: x.strip().split('\t')[1] for x in f.readlines()}

def load_vis_fea(vis_dir, mapping):
  vis_fea = {}
  with open(mapping, 'r') as f:
    for line in f:
      if '\t' not in line:
        l_id = line.strip()
        with open(vis_dir + '/' + l_id) as f_vis:
          vis_fea[l_id] = [x.strip() for x in f_vis.readlines()]
  return vis_fea

def load_wiki_concept(filename, stopwords, is_math=False):
  wiki_concept = {}
  count = 0
  with open(filename) as f:
    for line in f:
      items = line.strip().split('\t')
      for item in items:
        term = ' '.join(preprocess_content(item, stopwords, is_math))
        if term not in wiki_concept:
          wiki_concept[term] = []
        if count not in wiki_concept[term]:
          wiki_concept[term].append(count)
      count += 1
  return (wiki_concept, count)

def load_id_mapping(filename):
  id_mapping = {}
  counter = 0
  page_idx = 1
  l_id = ''
  with open(filename, 'r') as f:
    for line in f:
      if '\t' in line:
        id_mapping['s' + str(counter)] = {'l_id': l_id, 'page_idx': page_idx}
        page_idx += 1
        counter += 1
      else:
        l_id = line.strip()
        page_idx = 1
  return id_mapping

def load_raw_data(filename):
  data = {}
  with open(filename, 'r') as f:
    for line in f:
      line = line.strip()
      if 'id' not in data:
        data['id'] = line
      elif 'link_1' not in data:
        data['link_1'] = line
      elif 'link_2' not in data:
        data['link_2'] = line
      elif 'content' not in data:
        data['content'] = line
      elif line:
        data['content'] += ' ' + line
      else:
        return_data = copy.deepcopy(data)
        data = {}
        yield return_data

def preprocess_content(content, stopwords, is_math=False):
  # remove unicode
  content = content.decode('unicode_escape').encode('ascii','ignore')
  content = re.sub('-', ' ', content)
  # map simple equation to tokens
  if is_math:
    content = simple_eq_to_text(content)

  # remove punctuation
  content = "".join(l for l in content if l not in string.punctuation)
  content = content.lower()
  content = re.sub('\s+', ' ', content)
  words = re.split(' ', content)

  # Remove some slide tags -> new in TFIDF_math
  slide_tags = ['rrb', 'lrb', 'rsb', 'lsb']
  words = filter(lambda word: word not in slide_tags, words)
  # Remove stopwords
  words = filter(lambda word: word not in stopwords, words)
  # Stemming
  words = [stemmer.stem(word) for word in words]
  # remove digits
  if not is_math:
    words = filter(lambda word: not re.match('\d+$', word), words)
  return words

def simple_eq_to_text(content):
  content = re.sub('\s+', ' ', content.strip())
  # Use space to words -> add one space to prevent overlapping
  content = ' ' + re.sub(' ', '  ', content) + ' '
  pattern_mapping = {
    ' \+ ': ' plus ',
    # ' \xc3\x97 ': ' times ', cause unicode error
    ' \/ ': ' divide ',
    ' % ': ' percent ',
    ' = ': ' equal '
  }
  # map symbol to text
  for from_p, to_p in pattern_mapping.iteritems():
    content = re.sub(from_p, to_p, content)
  # map decimal to unique token
  content = re.sub(r' (\d+)\.(\d+) ', r' \1dot\2 ', content)
  # map fraction to unique token
  content = re.sub(r' (\d+)\/(\d+) ', r' \1divide\2 ', content)
  # remove added space
  content = re.sub('\s+', ' ', content.strip())
  return content

def load_stopwords(filename):
  with open(filename) as f:
    return [x.strip() for x in f.readlines()]

