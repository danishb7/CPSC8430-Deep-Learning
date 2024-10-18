# nltk installation for BLEU

pip install nltk

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Loading Captions and Feature Files

class VideoCaptionDataset(Dataset):
    def __init__(self, id_file, feat_folder, captions_data, vocab):
        self.feat_folder = feat_folder
        self.vocab = vocab
        
        # Load video IDs from id.txt
        with open(id_file, 'r') as f:
            self.video_ids = [line.strip() for line in f.readlines()]
        
        # Numericalize captions
        self.captions_data = {item['id']: self.vocab.numericalize(item['caption'][0]) for item in captions_data}

    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        # Get the video ID
        video_id = self.video_ids[idx]
        
        # Load the precomputed features from the .npy file
        feat_path = os.path.join(self.feat_folder, video_id + '.npy')
        features = np.load(feat_path)
        features = torch.tensor(features, dtype=torch.float32)
        
        # Get the corresponding numericalized caption and convert to tensor
        caption = torch.tensor(self.captions_data[video_id], dtype=torch.long)
        
        return features, caption


import torch.nn.functional as F

def pad_collate_fn(batch):

    
    batch_features, batch_captions = zip(*batch)
    
    # Stack the features
    batch_features = torch.stack(batch_features, dim=0)
    
    # Find the length of the longest caption
    max_length = max([len(caption) for caption in batch_captions])
    
    # Pad all captions to the same length with <PAD> token
    padded_captions = []
    for caption in batch_captions:
        padded_caption = F.pad(torch.tensor(caption), (0, max_length - len(caption)), value=vocab.word2idx[vocab.pad_token])
        padded_captions.append(padded_caption)
    
    # Stack padded captions
    padded_captions = torch.stack(padded_captions, dim=0)
    
    return batch_features, padded_captions



# Vocabulary class
from collections import Counter
import itertools

class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {}
        self.idx2word = {}
        self.min_freq = min_freq
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.word_counter = Counter()

    def build_vocab(self, captions):
        for caption in captions:
            tokens = caption.split() 
            self.word_counter.update(tokens)
        

        idx = 0
        for token in self.special_tokens:
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            idx += 1
        

        for word, count in self.word_counter.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def numericalize(self, text):

        return [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in text.split()]

    def denumericalize(self, indices):

        return [self.idx2word.get(idx, self.unk_token) for idx in indices]

# Extracts all captions from the JSON data
all_captions = list(itertools.chain.from_iterable([item['caption'] for item in train_captions]))

# Creates the vocabulary and build it from the captions
vocab = Vocabulary(min_freq=3)
vocab.build_vocab(all_captions)

# print(f"Vocabulary size: {len(vocab.word2idx)}")

all_captions = list(itertools.chain.from_iterable([item['caption'] for item in train_captions]))

# Creates the vocabulary and build it from the captions
vocab = Vocabulary(min_freq=3)
vocab.build_vocab(all_captions)

# print(f"Vocabulary size: {len(vocab.word2idx)}")


# Encoder
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(feature_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, features):
        # features: (batch_size, num_frames, feature_size)
        features = self.fc(features) 
        features = self.relu(features)
        return features

# Decoder
class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_size, num_layers=1):
        super(DecoderWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = nn.Linear(hidden_size, hidden_size)

    def forward(self, features, captions, hidden=None):
        batch_size = features.size(0)
        num_frames = features.size(1)
        hidden_size = features.size(2)
        

        if hidden is None:
            h_0 = torch.zeros(1, batch_size, hidden_size).to(features.device)  
            c_0 = torch.zeros(1, batch_size, hidden_size).to(features.device)  
            hidden = (h_0, c_0)
        
        # Embed the captions
        embeddings = self.embedding(captions)  
        
        outputs = []
        
        for t in range(embeddings.size(1)):
            # Apply attention over the video features
            attention_weights = torch.bmm(features, hidden[0][-1].unsqueeze(2)).squeeze(2) 
            attention_weights = torch.softmax(attention_weights, dim=1)
            attention_applied = torch.bmm(attention_weights.unsqueeze(1), features)  
            
            # Concatenate the attention-applied video features with the current word embedding
            lstm_input = torch.cat((attention_applied.squeeze(1), embeddings[:, t, :]), dim=1) 
            
            # Pass through LSTM
            lstm_output, hidden = self.lstm(lstm_input.unsqueeze(1), hidden)  
            
            # Generate the output (next word prediction)
            output = self.fc(lstm_output.squeeze(1))
            outputs.append(output)
        
        # Stack outputs along the time dimension
        outputs = torch.stack(outputs, dim=1) 
        
        return outputs, hidden


# Seq2Seq model

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features, captions):
        # Passes the features through the encoder
        encoder_outputs = self.encoder(features)
        
        # Passes the encoded features and captions to the decoder
        outputs, _ = self.decoder(encoder_outputs, captions) 
        
        return outputs


# train_id_file = './data/MLDS_hw2_1_data/training_data/id.txt'       # to be changed by the user while testing
# train_feat_folder = './data/MLDS_hw2_1_data/training_data/feat'     # to be changed by the user while testing
# train_captions_file = './data/MLDS_hw2_1_data/training_label.json'  # to be changed by the user while testing

# # Paths to the testing data
# test_id_file = './data/MLDS_hw2_1_data/testing_data/id.txt'         # to be changed by the user while testing
# test_feat_folder = './data/MLDS_hw2_1_data/testing_data/feat'       # to be changed by the user while testing
# test_captions_file = './data/MLDS_hw2_1_data/testing_label.json'    # to be changed by the user while testing


# Model training

# import torch.optim as optim
# import torch.nn as nn

# # Hyperparameters
# feature_size = 4096  # feature size
# hidden_size = 512    # size of hidden state in the LSTM
# vocab_size = len(vocab.word2idx)  # Size of the vocabulary
# embedding_size = 256  # Size of the word embeddings
# num_layers = 1  # Number of LSTM layers

# # Initialize encoder and decoder
# encoder = Encoder(feature_size, hidden_size)
# decoder = DecoderWithAttention(hidden_size, vocab_size, embedding_size, num_layers)

# # Create Seq2Seq model
# seq2seq_model = Seq2Seq(encoder, decoder)

# criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.pad_token])
# optimizer = optim.Adam(seq2seq_model.parameters(), lr=0.00005)

# num_epochs = 100
# seq2seq_model.train()

# for epoch in range(num_epochs):
#     for i, (features, captions) in enumerate(train_dataloader):
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = seq2seq_model(features, captions[:, :-1])
#         targets = captions[:, 1:].reshape(-1)
        
#         # flattens the outputs for loss calculation
#         outputs = outputs.view(-1, outputs.size(-1))
        
#         # computes the loss
#         loss = criterion(outputs, targets)
        
#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         if i % 10 == 0:
#             print(f'Epoch {epoch+1}/{num_epochs}, Step {i+1}, Loss: {loss.item()}')


# Model Evaluation
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# BLEU score calculation function
def calculate_bleu(reference, candidate):
    smooth = SmoothingFunction().method4
    return sentence_bleu([reference], candidate, smoothing_function=smooth)

# Function to evaluate the model and calculate BLEU scores
def evaluate_model(test_dataloader, seq2seq_model, vocab):
    seq2seq_model.eval()
    total_bleu_score = 0
    total_samples = 0

    with torch.no_grad():
        for features, captions in test_dataloader:
            # Forward pass through the model
            outputs = seq2seq_model(features, captions[:, :-1])
            _, predicted_indices = torch.max(outputs, dim=2)

            # Iterate over each sample in the batch
            for i in range(features.size(0)):

                predicted_caption = vocab.denumericalize(predicted_indices[i])
                reference_caption = vocab.denumericalize([idx for idx in captions[i].tolist() if idx != vocab.word2idx[vocab.pad_token]])

                # Calculates BLEU score
                bleu_score = calculate_bleu(reference_caption, predicted_caption)
                total_bleu_score += bleu_score
                total_samples += 1

    # Computes average BLEU score
    average_bleu_score = total_bleu_score / total_samples if total_samples > 0 else 0
    print(f"Average BLEU Score: {average_bleu_score}")





if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 seq2seq_model.py <data_directory> <output_filename>")
        sys.exit(1)

    data_directory = sys.argv[1]
    output_filename = sys.argv[2]

    # Paths to the training data
    train_id_file = './data/MLDS_hw2_1_data/training_data/id.txt'       # to be changed by the user while testing
    train_feat_folder = './data/MLDS_hw2_1_data/training_data/feat'     # to be changed by the user while testing
    train_captions_file = './data/MLDS_hw2_1_data/training_label.json'  # to be changed by the user while testing

    # Paths to the testing data
    test_id_file = './data/MLDS_hw2_1_data/testing_data/id.txt'         # to be changed by the user while testing
    test_feat_folder = './data/MLDS_hw2_1_data/testing_data/feat'       # to be changed by the user while testing
    test_captions_file = './data/MLDS_hw2_1_data/testing_label.json'    # to be changed by the user while testing

    # Load captions JSON file for training
    with open(train_captions_file, 'r') as f:
        train_captions = json.load(f)

    # Load captions JSON file for testing
    with open(test_captions_file, 'r') as f:
        test_captions = json.load(f)

    dataset = VideoCaptionDataset(id_file=train_id_file, feat_folder=train_feat_folder, captions_data=train_captions, vocab=vocab)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=pad_collate_fn)


    # # dataset and dataloader for training
    # train_dataset = VideoCaptionDataset(id_file=train_id_file, feat_folder=train_feat_folder, captions_data=train_captions, vocab=vocab)
    # train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=pad_collate_fn)

    # # dataset and dataloader for testing
    # test_dataset = VideoCaptionDataset(id_file=test_id_file, feat_folder=test_feat_folder, captions_data=test_captions, vocab=vocab)
    # test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=pad_collate_fn)

    # Initialize the Seq2Seq model
    feature_size = 4096
    hidden_size = 512
    embedding_size = 256
    vocab_size = len(vocab.word2idx)
    num_layers = 1

    encoder = Encoder(feature_size, hidden_size)
    decoder = DecoderWithAttention(hidden_size, vocab_size, embedding_size, num_layers)
    seq2seq_model = Seq2Seq(encoder, decoder)

    seq2seq_model.load_state_dict(torch.load('seq2seq_model.pth'))

    # Evaluate the model
    bleu_score = evaluate_model(dataloader, seq2seq_model, vocab)
    
    # Save the output to the specified file
    with open(output_filename, 'w') as f:
        f.write(f"Average BLEU Score: {bleu_score}\n")

    print(f"Results saved to {output_filename}")