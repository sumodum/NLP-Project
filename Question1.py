import pandas as pd
import os
import gensim.downloader
from gensim.models import Word2Vec
import numpy as np
from gensim.models import KeyedVectors
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score
import timeit
import _pickle as cPickle
import os
import sys
import codecs
import re
import numpy as np
from itertools import chain
import sklearn
from gensim.models import KeyedVectors
from gensim import models

#Constans and Datapaths
train_dir = './data/eng.train'
dev_dir =  './data/eng.testa'
test_dir =  './data/eng.testb'

#Part 1.1 
print("***Downloading Word2Vec Model***")
w2v = gensim.downloader.load("word2vec-google-news-300")
w2v.save("word2vec.wordvectors")

#if already have this file!
w2v = KeyedVectors.load("word2vec.wordvectors", mmap='r')

print("*** Part 1.1 ***")
for w in ["student","Apple","apple"]:
    best = w2v.most_similar(positive=[w])
    print("Most Similar Word to "+ w + " : " + str(best[0][0]) + " with a cosine similarity of " + str(best[0][1]))

#Part 1.2 

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

train_sentences = load_sentences(train_dir, True)
test_sentences = load_sentences(test_dir, True)
dev_sentences = load_sentences(dev_dir, True)

def get_unique_tags(sentences): 
    unique_tags = []
    for sentence in sentences: 
        for word in sentence: 
            if word[3] not in unique_tags:
                unique_tags.append(word[3])
    return unique_tags
print("*** Part 1.2 *** ")
print("Unique tags for training set:")
print(get_unique_tags(train_sentences))
            
print("Unique tags for dev set: ")
print(get_unique_tags(dev_sentences))

print("Unique tags for test set: ")  
print(get_unique_tags(test_sentences))

print("Number of sentences (training):", len(train_sentences))
print("Number of sentences (dev):", len(dev_sentences))
print("Number of sentences (test):", len(test_sentences))
#Multiple Named Entities
def get_sentence(sentences,sample_sentences):
    for sentence in sentences:
        entities = 0
        for i in range(0,len(sentence)-1):
            if ("B-" in sentence[i][3] and "I-" in sentence[i+1][3]): #named entity with more than 1 word 
                entities += 1
        if entities >= 2: 
            sample_sentences.append(sentence)
    return sample_sentences   

sample_sentences=[]
result = get_sentence(train_sentences,sample_sentences)
for sentence in result:
    print(sentence)
l = []
for word in result[0]:
    l.append(word[0])
result_sentence = " ".join(l)
print(result_sentence)

def get_named_entities(sentence):
    inside_tags = ['I-ORG', 'I-LOC', 'I-PER', 'I-MISC'] 
    begin_tags = ['B-LOC', 'B-ORG', 'B-MISC']
    outside_tags = ['O']
    entities = []
    entity = [] 
    for word in sentence: 
        if word[3] in begin_tags or word[3] in outside_tags and len(entity)!=0:
            entities.append(' '.join(entity))
            entity = []
        if word[3] in begin_tags or word[3] in inside_tags:
            entity.append(word[0])
    return entities
print("*** Extracting Named Entities from Sentence: ***")
print(get_named_entities(result[0]))


# Part 1.3 
print("*** Part 1.3 ***")
tags = (get_unique_tags(sorted(train_sentences)))
tag2idx = {k: v for v, k in enumerate(tags)}
idx2tag = {v: k for k, v in tag2idx.items()}


def extractWordAndLabel(dataset,tag2idx):
    dfx = []
    for s in dataset:
        for w in s: 
            dfx.append((w[0],tag2idx[w[-1]]))
    df = pd.DataFrame(dfx, columns=['word', 'label'])
    return df

train_df = extractWordAndLabel(train_sentences,tag2idx)
dev_df = extractWordAndLabel(dev_sentences,tag2idx)
test_df = extractWordAndLabel(test_sentences,tag2idx)
            

# embedding_dim = w2v.vector_size
embedding_dim = 300
np.random.seed(0)
unknown = np.random.rand(embedding_dim)

class NER(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size, w2v):
        super(NER, self).__init__()
        self.word2vec = w2v
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True,batch_first = True)
        self.reLU = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.outputLayer = nn.Linear(2 * hidden_dim, output_size)

    def forward(self, word):
        word_embeddings = [self.word2vec[word][0:50] if word in self.word2vec else unknown for word in word]
        word_embeddings_array = np.stack(word_embeddings)
        word_vectors = torch.tensor(word_embeddings_array).to(dtype=torch.float32)
        lstm_output, _ = self.lstm(word_vectors)
        lstm_output = self.reLU(lstm_output)
        lstm_output = self.dropout(lstm_output)
        output = self.outputLayer(lstm_output)
        return output

      

class NERDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.data.iloc[idx]['word']
        label = self.data.iloc[idx]['label']
        return word, torch.tensor(label)

train_dataset = NERDataset(train_df)
test_dataset = NERDataset(dev_df)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = NER(embedding_dim=embedding_dim, hidden_dim=64, output_size=8, w2v=w2v)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def calculate_f1_score(predictions, labels):
    return f1_score(labels, predictions, average='micro')

best_f1_score = 0.0
best_model_state = None
patience = 5
total_time = 0
run_model = True
if run_model:
    print("*** Training Phase using maximnum of 20 epochs: ***")
    for epoch in range(100):
        starttime = timeit.default_timer()
        model.train() 
        total_loss = 0.0

        for batch in train_dataloader:
            words, labels = batch
            tag_score = model(words)
            loss = loss_fn(tag_score.view(-1, model.outputLayer.out_features), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval() 
        dev_predictions = []
        dev_labels = []

        with torch.no_grad():
            for batch in test_dataloader:
                words, labels = batch
                tag_score = model(words)
                _, predicted = torch.max(tag_score, 1)
                dev_predictions.extend(predicted.view(-1).tolist())
                dev_labels.extend(labels.view(-1).tolist())

        f1_dev = calculate_f1_score(dev_predictions, dev_labels)
        time_taken = timeit.default_timer() - starttime
        total_time += time_taken
        print(f"Epoch {epoch + 1}: Loss={total_loss:.4f}, F1 Dev={f1_dev:.4f} Training Time={time_taken: .2f} seconds")
        if f1_dev > best_f1_score:
            best_f1_score = f1_dev
            best_model_state = model.state_dict()
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"No improvement for {patience} epochs. Training stopped.")
            break
print(f"Total time taken: {total_time: .2f} seconds")
if best_model_state is not None: 
    torch.save(best_model_state, 'Part1Model.pt')
    print("Trained model saved as Part1Model.pt")

model.load_state_dict(torch.load('Part1Model.pt'))
model.eval()

wordsOnly = []
tagsOnly = []
for sentence in test_sentences:
    words = []
    tags = []
    for word in sentence:
        words.append(word[0])
        tags.append(word[-1])
    wordsOnly.append(words)
    tagsOnly.append(tags)

results = []
for sentence in wordsOnly:
    predicted_labels = []
    test_output = model(sentence)
    _, predicted = torch.max(test_output, 1)
    predicted_labels = [idx2tag[i] for i in predicted.tolist()]
    results.append(predicted_labels)



flatten_results = list(chain.from_iterable(results))
flatten_true = list(chain.from_iterable(tagsOnly))

macro = sklearn.metrics.f1_score(flatten_true,flatten_results, average='macro')
micro = sklearn.metrics.f1_score(flatten_true,flatten_results, average='micro')
print(f"F1 Macro Score: {macro} , F1 Micro Score: {micro}")



#Analysis of Micro score vs Macro score
print("*** Further analysis of Micro vs Macro F1 score of our model ***")

def CountFrequency(my_list):
 
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return freq
            
            
true_dict = CountFrequency(flatten_true)
predict_dict = CountFrequency(flatten_results)

wrong = {}
correct = {}
for i in range(len(flatten_true)):
    true_tag = flatten_true[i]
    predict_tag = flatten_results[i]

    if true_tag != predict_tag:
        if true_tag not in wrong:
            wrong[true_tag] = 1 
        else:
            wrong[true_tag] += 1
    else: 
        if true_tag not in correct:
            correct[true_tag] = 1
        else:
            correct[true_tag] += 1

            wordsOnly = []
tagsOnly = []
for sentence in train_sentences:
    words = []
    tags = []
    for word in sentence:
        words.append(word[0])
        tags.append(word[-1])
    wordsOnly.append(words)
    tagsOnly.append(tags)
training_tags = list(chain.from_iterable(tagsOnly))

training_tags_counted = CountFrequency(training_tags)


print("Correct tags predicted:")
print(correct)

print("Wrong tags predicted:")
print(wrong)

print("Number of each tags in our training set:")
print(training_tags_counted)


# def CountFrequency(my_list):
#     freq = {}
#     for item in my_list:
#         if (item in freq):
#             freq[item] += 1
#         else:
#             freq[item] = 1
#     return freq
# tagsOnly = []
# for sentence in train_sentences:
#     tags = []
#     for word in sentence:
#         tags.append(word[3])
#     tagsOnly.append(tags)
# training_tags = list(chain.from_iterable(tagsOnly))
# training_tags_counted = CountFrequency(training_tags)
# print("Frequency of tags in training set: ")
# print(training_tags_counted)

# #match each tag and word to unique integer
# tags = (get_unique_tags(sorted(train_sentences)))
# tags.append("<PAD>")
# tag2idx = {k: v for v, k in enumerate(tags)}
# idx2tag = {v: k for k, v in tag2idx.items()}
# word2idx = {"<UNK>":0,"<PAD>":1}
# print(tag2idx)
# for sentence in train_sentences:
#     for word in sentence:
#         if word[0] not in word2idx:
#             word2idx[word[0]] = len(word2idx)

# print(tag2idx)
# print(idx2tag)
# pretrained_embeddings = w2v
# embedding_dim = pretrained_embeddings.vector_size
# # Initialize the embedding layer
# vocab_size = len(word2idx)  # Adjust based on your vocabulary size
# embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# # Initialize the embeddings for known words using pretrained embeddings
# pretrained_word_indices = []  # List to store indices of words in the pretrained embeddings
# word_embeddings = []

# # Iterate through your vocabulary and find indices and embeddings for known words
# for word, idx in word2idx.items():
#     if word in pretrained_embeddings:
#         pretrained_word_indices.append(idx)
#         word_embeddings.append(pretrained_embeddings[word])

# # Set the weights for known words in the embedding layer
# embedding_layer.weight.data[pretrained_word_indices] = torch.tensor(word_embeddings, dtype=torch.float32)
# # Initialize a separate tensor for random OOV embeddings
# num_oov_words = vocab_size - len(pretrained_word_indices)
# if num_oov_words > 0:
#     random_oov_embeddings = np.random.rand(num_oov_words, embedding_dim)
#     random_oov_embeddings = torch.tensor(random_oov_embeddings, dtype=torch.float32)

# # Concatenate the embeddings for known words with random OOV embeddings
# if num_oov_words > 0:
#     combined_embeddings = torch.cat([torch.tensor(word_embeddings, dtype=torch.float32), random_oov_embeddings])
#     # Set the weights in the embedding layer
#     embedding_layer.weight.data = combined_embeddings

# #freeze the pretrained embeddings
# embedding_layer.weight.requires_grad = False  


# def preprocess_data(sentences, word2idx, tag2idx):
#     input_data = [torch.tensor([word2idx.get(word[0], word2idx["<UNK>"]) for word in sentence]) for sentence in sentences]
#     target_data = [torch.tensor([tag2idx[word[-1]] for word in sentence]) for sentence in sentences]

#     # Padding
#     input_data = pad_sequence(input_data, batch_first=True, padding_value=word2idx["<PAD>"])
#     target_data = pad_sequence(target_data, batch_first=True, padding_value=tag2idx["<PAD>"])

#     return input_data, target_data

# input_data, target_data = preprocess_data(train_sentences,word2idx, tag2idx)
# input_dataset = TensorDataset(input_data, target_data)
# train_dataloader = DataLoader(input_dataset, batch_size=32, shuffle=True)

# dev_input_data, dev_target_data = preprocess_data(dev_sentences,word2idx, tag2idx)
# dev_dataset = TensorDataset(dev_input_data, dev_target_data)
# dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=True)

# class NERLSTM(nn.Module):
#     def __init__(self,embedding_layer, vocab_size, embedding_dim, hidden_dim, num_tags, dropout_prob):
#         super(NERLSTM, self).__init__()
#         # self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding = embedding_layer
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.relu = nn.ReLU()  # Add ReLU activation
#         self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer
#         self.fc = nn.Linear(2 * hidden_dim, num_tags)

#     def forward(self, x):
#         embedded = self.embedding(x)
#         lstm_out, _ = self.lstm(embedded)
#         lstm_out = self.dropout(lstm_out)  # Apply dropout
#         lstm_out = self.relu(lstm_out)  # Apply ReLU activation
#         logits = self.fc(lstm_out)
#         return logits

# ### PARAMETERS 
# vocab_size = len(word2idx)
# num_tags = len(tag2idx)

# embedding_dim = 300  # Adjust as needed
# hidden_dim = 64  # Adjust as needed
# dropout_prob = 0.1  # Adjust dropout probability as needed
# learning_rate = 0.001
# num_epochs = 30  # Adjust as needed
# patience = 3
# best_f1_score = 0.0
# no_improvement = 0
# best_model_state = None

# model = NERLSTM(embedding_layer,vocab_size, embedding_dim, hidden_dim, num_tags, dropout_prob)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()

# run_model = True #put false if loading model in 
# if run_model: 
#     print("Starting Training of LSTM Model")
#     for epoch in range(num_epochs):
#         starttime = timeit.default_timer()

#         model.train()
#         total_loss = 0.0

#         for batch in train_dataloader:
#             words, labels = batch
#             optimizer.zero_grad()
            
#             # Forward pass
#             tag_scores = model(words)
            
#             # Reshape tag_scores and labels for the loss function
#             tag_scores = tag_scores.view(-1, num_tags)
#             labels = labels.view(-1)
            
#             # Compute the loss
#             loss = criterion(tag_scores, labels)
            
#             # Backpropagation and optimization
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()

#         # Evaluate on the development set and measure F1 score
#         model.eval()
#         dev_predictions = []
#         dev_labels = []

#         with torch.no_grad():
#             for batch in dev_dataloader:
#                 words, labels = batch
#                 tag_scores = model(words)
#                 _, predicted = torch.max(tag_scores, 2)
#                 dev_predictions.extend(predicted.view(-1).tolist())
#                 dev_labels.extend(labels.view(-1).tolist())

#         f1_dev = f1_score(dev_labels, dev_predictions, average='micro')
#         time_taken = timeit.default_timer() - starttime
#         print(f"Epoch {epoch + 1}/{num_epochs}: Loss={total_loss:.4f}, F1 Dev={f1_dev:.4f}, Time Taken={time_taken:.2f}")
#         if f1_dev > best_f1_score:
#             best_f1_score = f1_dev
#             best_model_state = model.state_dict()
#             no_improvement = 0
#         else:
#             no_improvement += 1

#         if no_improvement >= patience:
#             print(f"No improvement for {patience} epochs. Training stopped.")
#             break
#     if best_model_state is not None:
#         torch.save(best_model_state, 'Part1Model.pt')
#         print("Trained model saved as Part1Model.pt")


# model.load_state_dict(torch.load('Part1Model.pt'))
# model.eval()
# test_input_data, test_target_data = preprocess_data(test_sentences, word2idx, tag2idx)

# test_dataset = TensorDataset(test_input_data, test_target_data)

# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False) 

# test_predictions = []
# test_labels = []

# with torch.no_grad():
#     for batch in test_dataloader:
#         words, labels = batch
#         tag_scores = model(words)
#         _, predicted = torch.max(tag_scores, 2)
#         test_predictions.extend(predicted.view(-1).tolist())
#         test_labels.extend(labels.view(-1).tolist())

# f1_test = f1_score(test_labels, test_predictions, average='micro')
# f1_test2 = f1_score(test_labels, test_predictions, average='macro')

# print(f"F1 Micro Test Score: {f1_test:.4f}")
# print(f"F1 Macro Test Score: {f1_test2:.4f}")

# #Analysis of Micro score vs Macro score
# print("*** Further analysis of Micro vs Macro F1 score of our model ***")
# predicted_labels = [idx2tag[i] for i in test_predictions]
# true_labels = [idx2tag[i] for i in test_labels]
# true_dict = CountFrequency(true_labels)
# predict_dict = CountFrequency(predicted_labels)

# print("Frequency of tags in test set: ")
# print(true_dict)
# print("Frequnecy of tags in predicted tags by model: ")
# print(predict_dict)
# wrong = {}
# correct = {}
# for i in range(len(true_labels)):
#     true_tag = true_labels[i]
#     predict_tag = predicted_labels[i]

#     if true_tag != predict_tag:
#         if true_tag not in wrong:
#             wrong[true_tag] = 1 
#         else:
#             wrong[true_tag] += 1
#     else: 
#         if true_tag not in correct:
#             correct[true_tag] = 1
#         else:
#             correct[true_tag] += 1


# print("Correct tags predicted:")
# print(correct)

# print("Wrong tags predicted:")
# print(wrong)

