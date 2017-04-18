import sys
from data_handler import DataHandler
from network import Network

LSTM_SIZE = 128
NUM_LAYERS = 3
LEARNING_RATE = 0.003
NAME = 'lstm_rnn'
BATCH_SIZE = 128
TIME_STEPS = 100
NUM_TRAIN_BATCHES = 2000

text_file = ''
ckpt_file = ''


def prepare_network():
  # Load the file and embed it in its vocab space
  data = DataHandler(text_file)
  data.prepare_for_training()
  # Create a network
  return  Network(data, LSTM_SIZE, NUM_LAYERS, LEARNING_RATE, NAME, ckpt_file)

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