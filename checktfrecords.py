import tensorflow as tf
from tqdm import tqdm
import os
import sys
import numpy as np
import logging
import tokenization

max_seq_length = 512
max_predictions_per_seq = 20
tf.enable_eager_execution()
name_to_features = {
    "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
    "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
    "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
    # Note: We keep this feature name `next_sentence_labels` to be
    # compatible with the original data created by lanzhzh@. However, in
    # the ALBERT case it does represent sentence_order_labels.
    "next_sentence_labels": tf.FixedLenFeature([1], tf.int64),
    "masked_lm_positions": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
    "masked_lm_ids": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
    "masked_lm_weights": tf.FixedLenFeature([max_predictions_per_seq], tf.float32)
}

tokenizer = tokenization.FullTokenizer("../30k/tokenizer.vocab", do_lower_case=False, spm_model_file="../30k/tokenizer.model")

def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)
  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example

BUCKET_NAME = "bertsweeu" #@param {type:"string"}
MODEL_DIR = "albert_test_plz" #@param {type:"string"}
PRETRAINING_DIR = "albert_data" #@param {type:"string"}

# Training procedure config
NUM_TPU_CORES = 8

if BUCKET_NAME:
  BUCKET_PATH = "gs://{}".format(BUCKET_NAME)
else:
  BUCKET_PATH = "."

BERT_GCS_DIR = "{}/{}".format(BUCKET_PATH, MODEL_DIR)
DATA_GCS_DIR = "{}/{}".format(BUCKET_PATH, PRETRAINING_DIR)
CONFIG_GCS_DIR = "{}/{}/{}".format(BUCKET_PATH, "albert_base", "albert_config.json")

input_files = tf.gfile.Glob(os.path.join("./test",'*tfrecord'))
input_file_cmd = ",".join(input_files)

input_files2 = input_files
print(len(input_files))
#cc458-2.txt.tfrecord
#cc459-1.txt.tfrecord
#cc464-1.txt.tfrecord
c = 0
for fn in tqdm(input_files2):

  for record in tf.python_io.tf_record_iterator(fn):
    c += 1
    example = _decode_record(record, name_to_features)
    print(tokenizer.convert_ids_to_tokens(example['input_ids'].numpy().tolist()))
    print(example['segment_ids'].numpy().tolist())
    print(example)


print("total:")
print(c)
