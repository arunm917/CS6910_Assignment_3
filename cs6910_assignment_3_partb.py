# -*- coding: utf-8 -*-
"""CS6910_Assignment_3_PartB

# Downloading necessary packages and files
"""

import gdown
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import StepLR
import wandb

wandb.login()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# downloading file from gdrive
output = 'tam_train'
file_id = '1pdJVD8P71fpqGRnvFfOp_6TbVft9NlnH' # Google drive ID
#Download the file
gdown.download('https://drive.google.com/uc?id=' + file_id, output, quiet=False)
print('DONE.')

# downloading file from gdrive
output = 'tam_valid'
file_id = '1pdp6ojHltRRNLXsmoQbGRc2Qn8X1EUJV' # Google drive ID
#Download the file
gdown.download('https://drive.google.com/uc?id=' + file_id, output, quiet=False)
print('DONE.')

# downloading file from gdrive
output = 'tam_test'
file_id = '1pdaTq-g2ZKhRKv6fRrSbEsJkOH5gdrEQ' # Google drive ID
#Download the file
gdown.download('https://drive.google.com/uc?id=' + file_id, output, quiet=False)
print('DONE.')

torch.manual_seed(55)
np.random.seed(55)

"""#Preprocessing"""

train_data_df = pd.read_csv('tam_train')
valid_data_df = pd.read_csv('tam_valid')
test_data_df = pd.read_csv('tam_test')

train_data_df.columns = ['English','Tamil']
valid_data_df.columns = ['English','Tamil']
test_data_df.columns = ['English','Tamil']

"""# Creating vocabulary and padding"""

# Checkign unique chars
############################################## Train data #########################################################
char_list_eng_train = []
for i in range(len(train_data_df['English'])):
  char = [*train_data_df.loc[i, 'English']]
  char_list_eng_train.extend(char)

char_list_tam_train = []
for i in range(len(train_data_df['Tamil'])):
  char = [*train_data_df.loc[i, 'Tamil']]
  char_list_tam_train.extend(char)

############################################## Validation data #########################################################
char_list_eng_val = []
for i in range(len(valid_data_df['English'])):
  char = [*valid_data_df.loc[i, 'English']]
  char_list_eng_val.extend(char)

char_list_tam_val = []
for i in range(len(valid_data_df['Tamil'])):
  char = [*valid_data_df.loc[i, 'Tamil']]
  char_list_tam_val.extend(char)

############################################## Test data #########################################################
char_list_eng_test = []
for i in range(len(test_data_df['English'])):
  char = [*test_data_df.loc[i, 'English']]
  char_list_eng_test.extend(char)

char_list_tam_test = []
for i in range(len(test_data_df['Tamil'])):
  char = [*test_data_df.loc[i, 'Tamil']]
  char_list_tam_test.extend(char)


unique_tam_char_train = list(set(char_list_tam_train))
unique_tam_char_val = list(set(char_list_tam_val))
unique_tam_char_test = list(set(char_list_tam_test))


# Indexing
SOS_token = '<SOS>'
EOS_token = '<EOS>'
PAD_token = '<PAD>'
UNK_token = '<UNK>'

vocabulary_eng = list(set(char_list_eng_train))
vocabulary_eng = [PAD_token] + [UNK_token] + [SOS_token] + [EOS_token] + vocabulary_eng 

vocabulary_tam = list(set(char_list_tam_train))
vocabulary_tam = [PAD_token] + [UNK_token] + [SOS_token] + [EOS_token] + vocabulary_tam

char_index_eng = {value: index for index, value in enumerate(vocabulary_eng)}
char_index_tam = {value: index for index, value in enumerate(vocabulary_tam)}

idx2char_eng = {value: key for key, value in char_index_eng.items()}
idx2char_tam = {value: key for key, value in char_index_tam.items()}

def tokenize_eng(word):
    chars = [*word]
    tokens_eng = [char_index_eng[char] if char in char_index_eng else 0 for char in chars]
    
    return tokens_eng

def tokenize_tam(word):
    chars = [*word]
    tokens_tam = [char_index_tam[char] if char in char_index_tam else 0 for char in chars]
    
    return tokens_tam

# Define the training pairs
training_pairs = train_data_df.values.tolist()
val_pairs = valid_data_df.values.tolist()
test_pairs = test_data_df.values.tolist()


eng_words = [tokenize_eng(pair[0]) for pair in training_pairs]
tam_words = [tokenize_tam(pair[1]) for pair in training_pairs]

# Determining max length english

lengths_eng = []
# max_length_eng = max([len(words) for words in eng_words])
for word in eng_words:

    word_length = len(word)
    lengths_eng.append(word_length)

max_length_tam = max([len(words) for words in tam_words])


# Determining max length english and tamil
max_length = max([len(words) for words in eng_words + tam_words])

def padding(word_pairs):
  ''' Function to pad the input and target sequences. Padding is done to ensure that
      all the training, validation and test samples are of equal size.'''
  
  eng_words = [tokenize_eng(pair[0]) for pair in word_pairs]
  tam_words = [tokenize_tam(pair[1]) for pair in word_pairs]

  
  padded_input_sequences = [torch.tensor([char_index_eng['<SOS>']] + eng_words + [char_index_eng['<EOS>']] + [(char_index_eng['<PAD>'])]*(max_length - len(eng_words))) for eng_words in eng_words]
  padded_target_sequences = [torch.tensor([char_index_eng['<SOS>']] + tam_words + [char_index_tam['<EOS>']] + [(char_index_tam['<PAD>'])]*(max_length - len(tam_words))) for tam_words in tam_words]
  tensor = torch.tensor([char_index_eng['<PAD>']]*(max_length+2))
  padded_input_sequences.append(tensor)
  padded_target_sequences.append(tensor)
  padded_input_sequences = torch.stack(padded_input_sequences)
  padded_target_sequences = torch.stack(padded_target_sequences)
  
  return(padded_input_sequences,padded_target_sequences)

# Creating datasets
training_input_sequences, training_target_sequences = padding(training_pairs)
train_dataset = torch.utils.data.TensorDataset(training_input_sequences, training_target_sequences)

val_input_sequences, val_target_sequences = padding(val_pairs)
val_dataset = torch.utils.data.TensorDataset(val_input_sequences, val_target_sequences)

test_input_sequences, test_target_sequences = padding(test_pairs)
test_dataset = torch.utils.data.TensorDataset(test_input_sequences, test_target_sequences)

"""# Architecture"""

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, bidirectionality, cell_type_encoder):
        super(Encoder, self).__init__()
        self.bidirectionality = bidirectionality

        if self.bidirectionality == 'YES':
          bidirectional = True
          self.directions = 2
        else:
          bidirectional = False
          self.directions = 1

        self.cell_type_encoder = cell_type_encoder
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)

        if self.cell_type_encoder == 'RNN':
          self.rnn = nn.RNN(embedding_size, hidden_size, num_layers,bidirectional = bidirectional, dropout=p)
        if self.cell_type_encoder == 'LSTM':
          self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers,bidirectional = bidirectional, dropout=p)
        if self.cell_type_encoder == 'GRU':
          self.rnn = nn.GRU(embedding_size, hidden_size, num_layers,bidirectional = bidirectional, dropout=p)

        self.fc_hidden = nn.Linear(hidden_size*self.directions, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*self.directions, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):


        embedding = self.dropout(self.embedding(x))

        if self.cell_type_encoder == 'LSTM':
          encoder_outputs, (hidden, cell) = self.rnn(embedding)
        else:
          encoder_outputs, hidden = self.rnn(embedding)

        if self.cell_type_encoder == 'LSTM':
          if self.bidirectionality == 'YES':
            row = 1
            hidden_list = []
            cell_list = []
            for i in range(hidden.shape[0]//2):
             
              hidden_concatenated = self.fc_hidden(torch.cat((hidden[row-1:row], hidden[row:row+1]), dim=2))
             
              cell_concatenated = self.fc_cell(torch.cat((cell[row-1:row], cell[row:row+1]), dim=2))
            
              hidden_list.append(hidden_concatenated)
              cell_list.append(cell_concatenated)
              row += 2

            hidden_tensor = torch.stack(hidden_list)
            cell_tensor = torch.stack(cell_list)
            hidden_squeezed = hidden_tensor.squeeze()
            cell_squeezed = cell_tensor.squeeze()
          else:
            hidden_squeezed = hidden
            cell_squeezed = cell
        else:
          if self.bidirectionality == 'YES':
            row = 1
            hidden_list = []
            for i in range(hidden.shape[0]//2):

              hidden_concatenated = self.fc_hidden(torch.cat((hidden[row-1:row], hidden[row:row+1]), dim=2))

              hidden_list.append(hidden_concatenated)
              row += 2

            hidden_tensor = torch.stack(hidden_list)
            hidden_squeezed = hidden_tensor.squeeze()
          else:
            hidden_squeezed = hidden


        if self.cell_type_encoder == 'LSTM':
          return encoder_outputs, hidden_squeezed, cell_squeezed
        else:
          return encoder_outputs, hidden_squeezed

class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p, cell_type_decoder, bidirectionality):
        super(Decoder, self).__init__()
    
        self.num_layers = num_layers
        self.bidirectionality = bidirectionality
        if self.bidirectionality == 'YES':
          bidirectional = True
          self.directions = 2
        else:
          bidirectional = False
          self.directions = 1
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type_decoder = cell_type_decoder

        self.embedding = nn.Embedding(input_size, embedding_size)
        if self.cell_type_decoder == 'RNN':
          self.rnn = nn.RNN((hidden_size*self.directions + embedding_size), hidden_size, self.num_layers, dropout=p)
        if self.cell_type_decoder == 'LSTM':
          self.rnn = nn.LSTM((hidden_size*self.directions + embedding_size), hidden_size, self.num_layers, dropout=p)
        if self.cell_type_decoder == 'GRU':
          self.rnn = nn.GRU((hidden_size*self.directions + embedding_size), hidden_size, self.num_layers, dropout=p)
        
        self.energy = nn.Linear(hidden_size*(self.directions + 1), 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim = 0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_outputs, hidden, cell = None):

        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))


        input_length = encoder_outputs.shape[0] # Decoder input is encoder output
        # print('input_length:', input_length)
        encoder_outputs_reshaped = encoder_outputs.repeat(self.num_layers,1,1)
        hidden_reshaped = hidden.repeat(input_length, 1, 1 ) # Reshaping decoder hidden so that it can be concatenated


        energy = self.relu(self.energy(torch.cat((hidden_reshaped, encoder_outputs_reshaped), dim = 2)))

        attention_scores = self.softmax(energy)
 
        context_vector = torch.einsum("snk,snl->knl", attention_scores, encoder_outputs_reshaped)
 
        rnn_input = torch.cat((context_vector, embedding), dim = 2)

        if self.cell_type_decoder == 'LSTM':
          outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        else:
          outputs, hidden = self.rnn(rnn_input, hidden)


        predictions = self.fc(outputs)


        predictions = predictions.squeeze(0)

        if self.cell_type_decoder == 'LSTM':
          return predictions, hidden, cell
        else:
          return predictions, hidden



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, cell_type_encoder, cell_type_decoder, bidirectionality, 
                 num_layers, hidden_size, teacher_forcing_ratio):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type_encoder = cell_type_encoder
        self.cell_type_decoder = cell_type_decoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.teacher_force_ratio = teacher_forcing_ratio
        
        if bidirectionality == 'YES':
          bidirectional = True
          self.directions = 2
        else:
          bidirectional = False
          self.directions = 1

    def forward(self, source, target):
        # cell = None
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(vocabulary_tam)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        if self.cell_type_encoder != 'LSTM':
          cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        if self.cell_type_encoder == 'LSTM':
          encoder_outputs, hidden, cell = self.encoder(source)
        else:
          encoder_outputs, hidden = self.encoder(source)

        x = target[0]
        predicted_sequences = torch.zeros([32, batch_size]).to(device)

        for t in range(1, target_len):
            if self.cell_type_decoder == 'LSTM':
              output, hidden, cell = self.decoder(x, encoder_outputs, hidden, cell)
            else:
              output, hidden = self.decoder(x, encoder_outputs, hidden)


            outputs[t] = output

            predicted_token = output.argmax(1)

            x = target[t] if random.random() < self.teacher_force_ratio else predicted_token

            predicted_sequences[t] = predicted_token.squeeze()
        
        predicted_sequences_copy = predicted_sequences[1:].t()

        target_copy = target[1:].t()
        correct_predictions_batch = correct_sequences_count(predicted_sequences_copy, target_copy)
        return outputs, correct_predictions_batch

def correct_sequences_count(predicted_sequences, target_sequences):
  
    correct_predictions_batch = 0
    for i in range(batch_size):

        target_word_list = []
        target_word_length = 0
        predicted_word_length = 0
        flag_target = 1
        flag_predicted = 1

        for element in target_sequences[i]:
            idx = element.item()
            target_char =  idx2char_tam[idx]
            target_word_list.append(target_char)
            if flag_target == 1:
              target_word_length += 1
              if idx == char_index_tam['<EOS>']:
                flag_target = 0
                break

        target_word_length = target_word_length - 1

    
        predicted_word_list = []
        for element in predicted_sequences[i]:
            idx = element.item()
            predicted_char =  idx2char_tam[idx]
            predicted_word_list.append(predicted_char)
            if flag_predicted == 1:
              predicted_word_length += 1
              if idx == char_index_tam['<EOS>']:
                flag_predicted = 0
                break
        
        predicted_word_length = predicted_word_length - 1

        
        if target_word_length == predicted_word_length:
          if all(x == y for x, y in zip(target_word_list, predicted_word_list)):
              correct_predictions_batch += 1

    return correct_predictions_batch

def accuracy(dataloader, model, optimizer, criterion):
  model.eval()

  with torch.no_grad():
    total_loss = 0
    correct_predictions_total = 0
    correct_predictions_batch = 0

    for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
        batch_idx += 1
        # Get input and targets and get to cuda
        inp_data = input_seq.t().to(device)
        target = target_seq.t().to(device)

        # Forward prop
        output, correct_predictions_batch = model(inp_data, target)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)
        total_loss += loss.item()
        correct_predictions_total += correct_predictions_batch
    eval_loss = total_loss/batch_idx
    accuracy = (correct_predictions_total/((batch_idx*batch_size) - 1))*100
    model.train()
  return eval_loss, accuracy

"""# Hyperparameters"""

# Training hyperparameters
batch_size = 256
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(vocabulary_eng)
input_size_decoder = len(vocabulary_tam)
output_size = len(vocabulary_tam)

"""# Training"""

# Creating Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

def train(num_epochs, model, optimizer, criterion, scheduler):
  for epoch in tqdm(range(num_epochs)):

      total_loss = 0
      correct_predictions_epoch = 0
      correct_predictions_batch = 0

      for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
          batch_idx += 1
          # Get input and targets and get to cuda
          inp_data = input_seq.t().to(device)
          target = target_seq.t().to(device)

          # Forward prop
          output, correct_predictions_batch = model(inp_data, target)
          
          output = output[1:].reshape(-1, output.shape[2])
          target = target[1:].reshape(-1)

          optimizer.zero_grad()
          loss = criterion(output, target)
          total_loss += loss.item()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
          optimizer.step()
          correct_predictions_epoch += correct_predictions_batch
      scheduler.step()
      loss_epoch = total_loss/batch_idx
      train_accuracy = (correct_predictions_epoch/((batch_idx*batch_size) - 1))*100
      val_loss, val_accuracy = accuracy(val_loader, model, optimizer, criterion)
      print('\nEpoc loss: %.4f' % loss_epoch, '\nCorrect predictions during epoch:',correct_predictions_epoch,
            '\nTraining accuracy: %.2f'% train_accuracy)
      print('\nValidation loss: %.4f'% val_loss, '\nValidation accuracy: %.2f'% val_accuracy)
      wandb.log({'loss_epoch': loss_epoch, 'Training accuracy':train_accuracy, 'Validation accuracy':val_accuracy})

  test_loss, test_accuracy = accuracy(test_loader, model, optimizer, criterion)
  print('Test loss: %.4f'% test_loss, '\nTest accuracy: %.2f'% test_accuracy)
  wandb.log({'Validation loss': val_loss,'Test loss': test_loss, 'Test accuracy':test_accuracy})

"""#WANDB"""

sweep_configuration = {
  'method': 'grid',
  'name': 'sweep_attention',
  'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
  'parameters': {
      'num_epochs':{'values':[20]},
      'learning_rate': {'values': [1e-3]},
      'weight_decay':{'values':[0.0001]},
      'embedding_size': {'values': [200]},
      'hidden_size': {'values': [1024]},
      'num_layers': {'values': [2]},
      'enc_dropout': {'values': [0.5]},
      'dec_dropout': {'values': [0.5]},
      'teacher_forcing_ratio':{'values':[0.5]},
      'bidirectionality': {'values': ['YES']},
      'cell_type_encoder':{'values': ['LSTM']},
      'cell_type_decoder': {'values': ['GRU']}
    } }

def wandbsweeps():
  wandb.init(project = 'CS6910_Assignment_3')
  wandb.run.name = (
        "lr"
        + str(wandb.config.learning_rate)
        + "hs"
        + str(wandb.config.hidden_size)
        + "ENCdr"
        + str(wandb.config.enc_dropout)
        + "DECdr"
        + str(wandb.config.dec_dropout)
        + "nl"
        + str(wandb.config.num_layers)
        + "ce"
        + str(wandb.config.cell_type_encoder)
        + "de"
        + str(wandb.config.cell_type_decoder)
    )

  encoder_net = Encoder(input_size_encoder, wandb.config.embedding_size, wandb.config.hidden_size,
                        wandb.config.num_layers, wandb.config.enc_dropout, wandb.config.bidirectionality,
                        wandb.config.cell_type_encoder).to(device)
  decoder_net = Decoder(input_size_decoder,wandb.config.embedding_size, wandb.config.hidden_size,
                        output_size, wandb.config.num_layers, wandb.config.dec_dropout,
                        wandb.config.cell_type_decoder, wandb.config.bidirectionality,).to(device)
  model = Seq2Seq(encoder_net, decoder_net, wandb.config.cell_type_encoder,
                  wandb.config.cell_type_decoder, wandb.config.bidirectionality,
                  wandb.config.num_layers, wandb.config.hidden_size, wandb.config.teacher_forcing_ratio).to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate, weight_decay = wandb.config.weight_decay)
  scheduler = StepLR(optimizer, step_size = 6, gamma = 0.5)

  pad_idx = char_index_eng['<PAD>']
  criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

  train(wandb.config.num_epochs, model, optimizer, criterion, scheduler)
  wandb.finish()



sweep_id = wandb.sweep(sweep= sweep_configuration, project = 'CS6910_Assignment_3')
wandb.agent(sweep_id, function = wandbsweeps)