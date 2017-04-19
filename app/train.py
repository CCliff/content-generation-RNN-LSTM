import sys
from data_handler import DataHandler
from network import Network

LSTM_SIZE = 128
NUM_LAYERS = 4
LEARNING_RATE = 0.003
NAME = 'lstm_rnn'
BATCH_SIZE = 128
TIME_STEPS = 50
NUM_TRAIN_BATCHES = 20000

OPT_DECAY = 0.99
OPT_MOMENTUM = 0.25
OPT_EPSILON = 1e-10
OPT_FORGET_BIAS = 5.0
OPT_STDDEV = 0.01

text_file = ''
ckpt_file = ''


def prepare_network():
  # Load the file and embed it in its vocab space
  data = DataHandler(text_file)
  data.prepare_for_training()
  # Create a network
  return  Network(data, LSTM_SIZE, NUM_LAYERS, LEARNING_RATE, OPT_DECAY, OPT_MOMENTUM, OPT_EPSILON, OPT_FORGET_BIAS, OPT_STDDEV, NAME, ckpt_file)

def train():
  network = prepare_network();
  # Train the network
  network.train_batch(BATCH_SIZE, NUM_TRAIN_BATCHES, TIME_STEPS)
  # Test the network 
  print network.get_sentence()

def get_sentence():
  network = prepare_network()

  print network.get_sentence()

# args = [{The file python is running}, {the command for the program}, {the text corprus}, {the network being used} ]

if len(sys.argv) >= 4:
  command   = sys.argv[1]
  text_file = sys.argv[2]
  ckpt_file = sys.argv[3]

if command == '--train':
  train()
elif command == '--sample':
  get_sentence()
else:
  print "Unknown command: either --train or --sample required"