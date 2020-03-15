import csv
import pandas as pd
train_dataset = pd.read_csv('Train_Data.csv')
toignore1=[i for i in range(1000,2000)]
toignore2=[i for i in range(22000,25000)]
toignore3=[i for i in range(27000,30000)]
toignore4=[i for i in range(45000,47000)]
to_ignore=toignore1+toignore2+toignore3+toignore4
train_dataset=train_dataset.drop(train_dataset.index[to_ignore])

train_dataset.to_csv('train_bert.csv', index=False, header=False)
csv.writer(open('train_bert.tsv', 'w+',encoding="utf-8"), delimiter='\t').writerows(csv.reader(open("train_bert.csv", encoding="utf-8")))
import warnings
warnings.filterwarnings('ignore')

import io
import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp
#from bert import data, model
np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
# change `ctx` to `mx.cpu()` if no GPU is available.
#ctx = mx.gpu(0)
# Automatically downloads and loads bert uncased model
bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
#print(bert_base)
import classification
# Attach a single classifier layer on top of language model

bert_classifier = classification.BERTClassifier(bert_base, num_classes=4, dropout=0.1)
# only need to initialize the classifier layer.
bert_classifier.classifier.initialize(init=mx.init.Normal(0.02))
bert_classifier.hybridize(static_alloc=True)

# softmax cross entropy loss for classification
loss_function = mx.gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=True)

metric = mx.metric.Accuracy()

tsv_file = io.open('train_bert.tsv', encoding='utf-8', newline='\r\n')
for i in range(5):
    print(tsv_file.readline())
# Modify newline parameter to support news articles loading which contains windows type newlines
class modifyread(nlp.data.TSVDataset):
    def _read(self):
        all_samples = []
        for filename in self._filenames:
            with io.open(filename, 'r', encoding=self._encoding, newline='\r\n') as fin:
                content = fin.read()
            samples = (s for s in self._sample_splitter(content) if not self._should_discard())
            if self._field_separator:
                if not self._allow_missing:
                    samples = [self._field_selector(self._field_separator(s)) for s in samples]
                else:
                    selected_samples = []
                    num_missing = 0
                    for s in samples:
                        try:
                            fields = self._field_separator(s)
                            selected_samples.append(self._field_selector(fields))
                        except IndexError:
                            num_missing += 1
                    if num_missing > 0:
                        warnings.warn('%d incomplete samples in %s'%(num_missing, filename))
                    samples = selected_samples
            all_samples += samples
        return all_samples
# TO skip the first line, in case of headers, change the value to 1 below
num_discard_samples = 0

# Split fields by tabs
field_separator = nlp.data.Splitter('\t')

data_train_raw = modifyread(filename='train_bert.tsv', sample_splitter=nlp.data.utils.Splitter('\r\n'),
                                 field_separator=field_separator,
                                 num_discard_samples=num_discard_samples,
                                 field_indices=None)
sample_id = 231
# Headline
#print(data_train_raw[sample_id][0])
# Articles
#print(data_train_raw[sample_id][1])
# Stance
#print(data_train_raw[sample_id][2])
import transform
# Use the vocabulary from pre-trained model for tokenization
bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)

# The maximum length of an input sequence
max_len = 200

# The labels for the four classes
all_labels = ["agree", "disagree", "discuss", "unrelated"]

# Transform the data as sentence pairs.
pair = True
transform = transform.BERTDatasetTransform(bert_tokenizer, max_len,
                                                class_labels=all_labels,
                                                has_label=True,
                                                pad=True,
                                                pair=pair)
data_train = data_train_raw.transform(transform)

print('vocabulary used for tokenization = \n%s'%vocabulary)
print('%s token id = %s'%(vocabulary.padding_token, vocabulary[vocabulary.padding_token]))
print('%s token id = %s'%(vocabulary.cls_token, vocabulary[vocabulary.cls_token]))
print('%s token id = %s'%(vocabulary.sep_token, vocabulary[vocabulary.sep_token]))
print('token ids = \n%s'%data_train[sample_id][0])
print('valid length = \n%s'%data_train[sample_id][1])
print('segment ids = \n%s'%data_train[sample_id][2])
print('label = \n%s'%data_train[sample_id][3])

# The hyperparameters
batch_size = 20
lr = 5e-6

# The FixedBucketSampler and the DataLoader for making the mini-batches
train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[1]) for item in data_train],
                                            batch_size=batch_size,
                                            shuffle=True)
bert_dataloader = mx.gluon.data.DataLoader(data_train, batch_sampler=train_sampler)

trainer = mx.gluon.Trainer(bert_classifier.collect_params(), 'adam',
                           {'learning_rate': lr, 'epsilon': 1e-9})

# Collect all differentiable parameters
# `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
# The gradients for these params are clipped later
params = [p for p in bert_classifier.collect_params().values() if p.grad_req != 'null']
grad_clip = 1bert_dataloader = mx.gluon.data.DataLoader(data_train, batch_sampler=train_sampler)

trainer = mx.gluon.Trainer(bert_classifier.collect_params(), 'adam',
                           {'learning_rate': lr, 'epsilon': 1e-9})

# Collect all differentiable parameters
# `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
# The gradients for these params are clipped later
params = [p for p in bert_classifier.collect_params().values() if p.grad_req != 'null']
grad_clip = 1



# Training the model with only two epochs to avoid overfitting
log_interval = 4
num_epochs = 2
for epoch_id in range(num_epochs):
    metric.reset()
    step_loss = 0
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(bert_dataloader):
        with mx.autograd.record():

            ''' Load the data to the GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)
'''
            # Forward computation
            out = bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))
            ls = loss_function(out, label).mean()

        # And backwards computation
        ls.backward()

        # Gradient clipping
        trainer.allreduce_grads()
        nlp.utils.clip_grad_global_norm(params, 1)
        trainer.update(1)

        step_loss += ls.asscalar()
        metric.update([label], [out])

        # Printing vital information
        if (batch_id + 1) % (log_interval) == 0:
            print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                         .format(epoch_id, batch_id + 1, len(bert_dataloader),
                                 step_loss / log_interval,
                                 trainer.learning_rate, metric.get()[1]))
            step_loss = 0
bert_classifier.save_parameters('parameters_bert')
