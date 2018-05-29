DEBUG = False

# hyperparameters
WINDOW = 300
STEP = 8 if DEBUG else 1

LSTM_SIZE = 4 if DEBUG else 64

LEARN_RATE = 0.1

BATCH_SIZE = 1000
EPOCH_COUNT = 2 if DEBUG else 100

DROPOUT = 0.9

# input/output
INPUT_DIR = 'data/'
OUTPUT_DIR = 'debug/' if DEBUG else 'output/'

def ConfigSummary():
    return {
        'window': WINDOW,
        'step': STEP,
        'lstm_size': LSTM_SIZE,
        'learning_rate': LEARN_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCH_COUNT,
    }
