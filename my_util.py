#! /usr/bin/env python

import copy
#import json
import math
import re
import string
import subprocess
import sys
#import unicodedata
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
from nltk.tag import StanfordPOSTagger

def modify_sparse(a, del_start, del_end, multi_start=None, multi_end=None, weight=None):
  for key in a.keys():
    if key >= del_start and key <= del_end:
      del a[key]
    if weight and key >= multi_start and key <= multi_end:
      a[key] *= weight
  return a

def cos_sim(a, b, vec):
  return (sum(sparse_to_vec(sparse_dot_mul(a, b), vec[:]))
    / math.sqrt(sum(sparse_to_vec(sparse_dot_mul(a, a), vec[:])))
    / math.sqrt(sum(sparse_to_vec(sparse_dot_mul(b, b), vec[:]))))

def cos_sim_smoothing(a, b, vec, smoothing):
  return (sum(sparse_dot_mul_to_vec_smoothing(a, b, vec[:], smoothing))
    / math.sqrt(sum(sparse_dot_mul_to_vec_smoothing(a, a, vec[:], smoothing)))
    / math.sqrt(sum(sparse_dot_mul_to_vec_smoothing(b, b, vec[:], smoothing))))

def sparse_dot_mul_to_vec_smoothing(a, b, vec, smoothing):
  for sparse in [a, b]:
    for key, value in sparse.iteritems():
      if key < len(vec):
        vec[key] *= value / smoothing
  return vec

def sparse_dot_mul(a, b):
  results = {}
  for key, value in a.iteritems():
    if key in b:
      results[key] = value * b[key]
  return results

def sparse_add(a, b):
  results = copy.deepcopy(a)
  for key, value in b.iteritems():
    if key in results:
      results[key] += value
    else:
      results[key] = value
  return results

def sparse_to_string(sparse, vec):
  return vec_to_string(sparse_to_vec(sparse, vec))

def vec_to_string(vec):
  return ' '.join(str(x) for x in vec)

def sparse_to_vec(sparse, vec):
  for w_id, w_score in sparse.iteritems():
    if w_id < len(vec):
      vec[w_id] = w_score
  return vec

def str_to_sparse(fea_str):
  return {int(x.split(':')[0]): float(x.split(':')[1]) for x in fea_str.split(' ')}

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
  elif resource_type == 'tx':
    f_o.write(
      ' '.join([key + ':' + str(value) for key, value in fea.iteritems()])
      + '\t'
      + page_idx
      + '\n'
    )

def load_dictionary_idf(filename):
  dictionary = load_dictionary(filename)
  with open(filename) as f:
    idf = {x.strip().split('\t')[1]: x.strip().split('\t')[2] for x in f.readlines()}
  return (dictionary, idf)

def load_dictionary(filename):
  with open(filename) as f:
    return {x.strip().split('\t')[0]: x.strip().split('\t')[1] for x in f.readlines()}

def get_txid(txid_mapping, tx_keys):
  txids = []
  if tx_keys:
    for tx_key in tx_keys.split(', '):
      txids.append(re.match('C(\d+_\d+).html', txid_mapping[tx_key]['l_id']).group(1))
  else:
    txids.append('0')
  return ' '.join(txids)

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

def load_id_mapping(filename, resource_type=None, tx_start_type=None):
  id_mapping = {}
  counter = 0
  page_idx = 1
  l_id = ''
  if resource_type and resource_type == 'tx':
    with open(filename, 'r') as f:
      for line in f:
        section = line.strip().split('\t')[0]
        if tx_start_type and tx_start_type == 'Ch':
          id_mapping[resource_type + str(counter)] = {
            'l_id': 'obs' + re.match('C(\d+)_(\d+).html', section).group(1),
            'page_idx': re.match('C(\d+_\d+).html', section).group(1)
          }
        else:
          id_mapping[resource_type + str(counter)] = {'l_id': section}
        counter += 1
  else:
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

def tokenization(f_pre_tokenized_name):
  documentPreprocessor = 'java edu.stanford.nlp.process.DocumentPreprocessor '
  return subprocess.Popen(documentPreprocessor + f_pre_tokenized_name, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def get_contaminated_tag_results(sentence, tagger):
  try:
    return tagger.tag(re.split('\s+', sentence))
  except OSError:
    return [(word, 'Not_defined') for word in re.split('\s+', sentence)]

def get_pos_tags(content, stopwords, is_stemming, is_math):
  # Content should be tokenized
  pos_tagger_dir = '/usr/users/swli/program/nlp_util/stanford-postagger'
  model = pos_tagger_dir + '/models/wsj-0-18-bidirectional-distsim.tagger'
  classpath = pos_tagger_dir + '/stanford-postagger_with_slf4j.jar'
  tagger = StanfordPOSTagger(model, classpath, java_options='-mx4000m')
  try:
    tag_results = tagger.tag(re.split('\s+', content))
  except OSError:
    sentences = re.split('\s+\.\s+', content)
    tag_results = []
    for index in range(len(sentences)):
      sentence = sentences[index]
      if index < len(sentences)-1:
        sentence += ' .'
      tag_results += get_contaminated_tag_results(sentence, tagger)
 
  pos_tags = []
  for pair in tag_results:
    word = pair[0]
    # map simple equation to tokens
    if is_math:
      word = simple_eq_to_text(word)
    # remove punctuation
    word = "".join(l for l in word if l not in string.punctuation)
    word = word.lower()
    word = process_word(word, stopwords, is_stemming, is_math)
    if word:
      pos_tags.append(pair[1])
  return pos_tags

def preprocess_content(content, stopwords, is_math=False, is_stemming=True, is_return_pos=False, is_return_math_words=False):
  pre_content = content
  # remove unicode
  # content = content.decode('unicode_escape').encode('ascii','ignore')
  content = "".join(map(lambda x: x if x in string.printable else ' ', content))

  content = re.sub('-', ' ', content)
  # Get pos tags
  if is_return_pos:
    pos_tags = get_pos_tags(content, stopwords, is_stemming, is_math)
  if is_return_math_words:
    math_words = get_math_words(content)
  # map simple equation to tokens
  if is_math:
    content = simple_eq_to_text(content)
  # remove punctuation
  content = "".join(l for l in content if l not in string.punctuation)
  content = content.lower()
  content = re.sub('\s+', ' ', content)

  words = [process_word(word, stopwords, is_stemming, is_math) for word in re.split(' ', content)]
  words = filter(lambda word: len(word) > 0, words)
  if not is_return_pos and not is_return_math_words:
    return words
  else:
    preprocess_result = (words,)
    if is_return_pos:
      if len(words) != len(pos_tags):
        print words
        print pos_tags
        print pre_content
        print content
        raise ValueError
      preprocess_result = preprocess_result + (pos_tags,)
    else:
      preprocess_result = preprocess_result + (None,)

    if is_return_math_words:
      preprocess_result = preprocess_result + (math_words,)
    else:
      preprocess_result = preprocess_result + (None,)

    return preprocess_result

def process_word(word, stopwords, is_stemming, is_math):
  # Remove stopwords
  # Remove some slide tags -> new in TFIDF_math
  slide_tags = ['rrb', 'lrb', 'rsb', 'lsb']
  if word in stopwords or word in slide_tags:
    return ''
  # Stemming
  if is_stemming:
    word = stemmer.stem(word)
  # remove digits
  if not is_math:
    return '' if re.match('\d+$', word) else word
  else:
    return '' if re.match(r'\d{8,}$', word) else word

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

def get_math_words(content):
  math_words = []
  content = re.sub('\s+', ' ', content.strip())
  content = ' ' + re.sub(' ', '  ', content) + ' '
  # map decimal to unique token
  content = re.sub(r' (\d+)\.(\d+) ', r' \1dot\2 ', content)
  # map fraction to unique token
  content = re.sub(r' (\d+)\/(\d+) ', r' \1divide\2 ', content)
  content = re.sub('\s+', ' ', content.strip())
  for word in re.split(' ', content):
    if (word in ['+', '/', '%', '=']
         or re.match('\d+dot\d+$', word)
         or re.match('\d+divide\d+$', word)
         or re.match(r'\d{1,7}$', word)
       ):
      math_words.append(word)
  return math_words

def load_stopwords(filename):
  with open(filename) as f:
    return [x.strip() for x in f.readlines()]

def print_vocab_list(filename, vocabs):
  vocabs = list(set(vocabs))
  f_o = open(filename, 'w')
  for vocab in vocabs:
    f_o.write('%s\n' % vocab)
  f_o.close()

