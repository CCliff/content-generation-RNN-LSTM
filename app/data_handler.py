import numpy as np

DEFAULT_PATH = 'data/shakespeare.txt'

class DataHandler:

  def __init__(self, ckpt_path):
    self.ckpt_path = ckpt_path
    self.data = []
    self.vocab = []
    self.embedded_data = []

  def get_data(self):
    return self.data

  def get_embedded_data(self):
    return self.embedded_data

  def get_vocab(self):
    return self.vocab or list(set(self.data)) 

  def load(self):
    data = ''
    if not self.ckpt_path:
    	self.ckpt_path = DEFAULT_PATH
    with open(self.ckpt_path, 'r') as f:
    	data += f.read()
    self.data = data.lower()
    
    return self.data

  def prepare_for_training(self):
    self.load()
    self.get_vocab()
    self.embedded_data = self.embed(self.get_data())
  
  def embed(self, text):

    # One hot encoding the text into a list 
    self.vocab = self.get_vocab()
    embedded_data = np.zeros((len(text), len(self.get_vocab())))

    cnt = 0 
    for char in text:
      v = [0.0]*len(self.get_vocab())
      v[self.vocab.index(char)] = 1.0
      embedded_data[cnt, :] = v
      cnt += 1

    return embedded_data

  def get_char_from_embed(self, index):
    return self.get_vocab()[index]