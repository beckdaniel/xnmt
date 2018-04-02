from xnmt.serialize.serializable import Serializable
from collections import defaultdict

class Vocab(Serializable):
  '''
  Converts between strings and integer ids.
  
  Configured via either i2w or vocab_file (mutually exclusive).
  
  Args:
    i2w (list of string): list of words, including <s> and </s>
    vocab_file (str): file containing one word per line, and not containing <s>, </s>, <unk>
  '''

  yaml_tag = "!Vocab"

  SS = 0
  ES = 1

  SS_STR = "<s>"
  ES_STR = "</s>"
  UNK_STR = "<unk>"

  def __init__(self, i2w=None, vocab_file=None):
    assert i2w is None or vocab_file is None
    if vocab_file:
      i2w = Vocab.i2w_from_vocab_file(vocab_file)
    if (i2w is not None):
      self.i2w = i2w
      self.w2i = {word: word_id for (word_id, word) in enumerate(self.i2w)}
      self.frozen = True
    else :
      self.w2i = {}
      self.i2w = []
      self.unk_token = None
      self.w2i[self.SS_STR] = self.SS
      self.w2i[self.ES_STR] = self.ES
      self.i2w.append(self.SS_STR)
      self.i2w.append(self.ES_STR)
      self.frozen = False
    self.overwrite_serialize_param("i2w", self.i2w)
    self.overwrite_serialize_param("vocab_file", None)

  @staticmethod
  def i2w_from_vocab_file(vocab_file):
    """
    Args:
      vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    """
    vocab = [Vocab.SS_STR, Vocab.ES_STR]
    reserved = set([Vocab.SS_STR, Vocab.ES_STR, Vocab.UNK_STR])
    with open(vocab_file, encoding='utf-8') as f:
      for line in f:
        word = line.strip()
        if word in reserved:
          raise RuntimeError(f"Vocab file {vocab_file} contains a reserved word: {word}")
        vocab.append(word)
    return vocab

  def convert(self, w):
    if w not in self.w2i:
      if self.frozen:
        assert self.unk_token != None, 'Attempt to convert an OOV in a frozen vocabulary with no UNK token set'
        return self.unk_token
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    return self.w2i[w]

  def __getitem__(self, i):
    return self.i2w[i]

  def __len__(self):
    return len(self.i2w)

  def freeze(self):
    """
    Mark this vocab as fixed, so no further words can be added. Only after freezing can the unknown word token be set.
    """
    self.frozen = True

  def set_unk(self, w):
    """
    Sets the unknown word token. Can only be invoked after calling freeze().
    
    Args:
      w (str): unknown word token
    """
    assert self.frozen, 'Attempt to call set_unk on a non-frozen dict'
    if w not in self.w2i:
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    self.unk_token = self.w2i[w]

class RuleVocab(Serializable):
  '''
  Converts between strings and integer ids
  '''

  yaml_tag = "!RuleVocab"

  SS = 0
  ES = 1

  SS_STR = u"<s>"
  ES_STR = u"</s>"
  UNK_STR = u"<unk>"

  def __init__(self, i2w=None, vocab_file=None):
    """
    :param i2w: list of words, including <s> and </s>
    :param vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    i2w and vocab_file are mutually exclusive
    """
    assert i2w is None or vocab_file is None
    self.tag_vocab = Vocab()
    self.lhs_to_index = defaultdict(list)

    if vocab_file:
      i2w = RuleVocab.i2w_from_vocab_file(vocab_file)
    if (i2w is not None):
      self.i2w = i2w
      self.w2i = {}
      for (word_id, word) in enumerate(self.i2w):
        self.w2i[word] = word_id
        if hasattr(word, 'lhs'):
          self.lhs_to_index[word.lhs].append(word_id)
          self.tag_vocab.convert(word.lhs)
          for r in word.open_nonterms:
            self.tag_vocab.convert(r)
    else :
      self.w2i = {}
      self.i2w = []
      self.unk_token = None
      self.w2i[self.SS_STR] = self.SS
      self.w2i[self.ES_STR] = self.ES
      self.i2w.append(self.SS_STR)
      self.i2w.append(self.ES_STR)

    self.frozen = False

    self.serialize_params = {"i2w": self.i2w}

  def freeze(self):
    self.frozen = True

  @staticmethod
  def i2w_from_vocab_file(vocab_file):
    """
    :param vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    """
    vocab = [Vocab.SS_STR, Vocab.ES_STR]
    reserved = set([Vocab.SS_STR, Vocab.ES_STR, Vocab.UNK_STR])
    with open(vocab_file, encoding='utf-8') as f:
      for line in f:
        word = line.strip()
        if word in reserved:
          raise RuntimeError(f"Vocab file {vocab_file} contains a reserved word: {word}")
        rule = Rule.from_str(word)
        vocab.append(rule)
    return vocab

  def convert(self, w):
    ''' w is a Rule object'''
    if w not in self.w2i:
      if self.frozen:
        assert self.unk_token != None, 'Attempt to convert an OOV in a frozen vocabulary with no UNK token set'
        return self.unk_token
      self.w2i[w] = len(self.i2w)
      self.lhs_to_index[w.lhs].append(len(self.i2w))
      self.i2w.append(w)

    if not self.frozen:
      self.tag_vocab.convert(w.lhs)
      for r in w.open_nonterms:
        self.tag_vocab.convert(r)

    return self.w2i[w]

  def rule_index_with_lhs(self, lhs):
    return self.lhs_to_index[lhs]

  def __getitem__(self, i):
    return self.i2w[i]

  def __len__(self):
    return len(self.i2w)

  def set_unk(self, w):
    assert self.frozen, 'Attempt to call set_unk on a non-frozen dict'
    if w not in self.w2i:
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    self.unk_token = self.w2i[w]


class Rule(Serializable):
  yaml_tag = "!Rule"

  def __init__(self, lhs, rhs=[], open_nonterms=[]):
    self.lhs = lhs
    self.rhs = rhs
    self.open_nonterms = open_nonterms
    self.serialize_params = {'lhs': self.lhs, 'rhs': self.rhs, 'open_nonterms': self.open_nonterms}

  def __str__(self):
    return (self.lhs + '|||' + ' '.join(self.rhs) + '|||' + ' '.join(self.open_nonterms))

  @staticmethod
  def from_str(line):
    segs = line.split('|||')
    assert len(segs) == 3
    lhs = segs[0]
    rhs = segs[1].split()
    open_nonterms = segs[2].split()
    return Rule(lhs, rhs, open_nonterms)

  def __hash__(self):
    #return hash(str(self) + " ".join(open_nonterms))
    if not hasattr(self, 'lhs'):
      return id(self)
    else:
      return hash(str(self))

  def __eq__(self, other):
    if not hasattr(other, 'lhs'):
      return False
    if not self.lhs == other.lhs:
      return False
    if not " ".join(self.rhs) == " ".join(other.rhs):
      return False
    if not " ".join(self.open_nonterms) == " ".join(other.open_nonterms):
      return False
    return True