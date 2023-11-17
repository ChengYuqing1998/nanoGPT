import os
from tqdm import tqdm
import numpy as np
# import tiktoken
from datasets import load_dataset # huggingface datasets
import pickle

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset("/Users/chengyuqing/Desktop/nanoGPT/openwebtext")

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val
    t = "".join(split_dataset['train']['text'][:1])
    # v = "".join(split_dataset['val']['text'][:240])
    v = "".join(split_dataset['train']['text'][160000:200000])
    # raise()
    # # train_list = []
    # # val_list = []
    char = ""
    # train_fea = ""
    # valid_fea = ""
    # train_len = len(split_dataset['train']['text'])
    # valid_len = len(split_dataset['val']['text'])
    # t = "".join(split_dataset['train']['text'][:20000])
    # for i in tqdm(range(train_len)):
    #     train_fea +=split_dataset['train']['text'][i]
    char_list = sorted(list(set(t)))
    # for j in tqdm(range(valid_len)):
    #     valid_fea += split_dataset['val']['text'][i]
    vocab_size = len(char_list)
    # print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")
    with open("/Users/chengyuqing/Desktop/nanoGPT/data/shakespeare_char/meta.pkl", 'rb') as f:
        meta = pickle.load(f)
    #meta = pickle.loads("/Users/chengyuqing/Desktop/nanoGPT/data/shakespeare_char/meta.pkl")
    # print(meta['itos'])
    # raise()

    # create a mapping from characters to integers
    stoi = meta['stoi']
        # {ch: i for i, ch in enumerate(char_list)}
    itos = meta['itos']
        # {i: ch for i, ch in enumerate(char_list)}

    def encode(s):
        enc_list = []
        for c in s:
            try:
                enc_list.append(stoi[c])
            except:
                continue
        return enc_list # encoder: take a string, output a list of integers


    def decode(l):
        dec_list = []
        for i in l:
            try:
                dec_list.append(itos[i])
            except:
                continue
        return ''.join(dec_list)  # decoder: take a list of integers, output a string

    train_ids = encode(t)
    val_ids = encode(v)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)


        # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    # enc = tiktoken.get_encoding("gpt2")
    # def process(example):
    #     ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    #     ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    #     # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    #     out = {'ids': ids, 'len': len(ids)}
    #     return out
    #
    # # tokenize the dataset
    # tokenized = split_dataset.map(
    #     process,
    #     remove_columns=['text'],
    #     desc="tokenizing the splits",
    #     num_proc=num_proc,
    # )
    #
    # # concatenate all the ids in each dataset into one large file we can use for training
    # for split, dset in tokenized.items():
    #     arr_len = np.sum(dset['len'], dtype=np.uint64)
    #     filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    #     dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    #     arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    #     total_batches = 1024
    #
    #     idx = 0
    #     for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
    #         # Batch together samples for faster write
    #         batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
    #         arr_batch = np.concatenate(batch['ids'])
    #         # Write into mmap
    #         arr[idx : idx + len(arr_batch)] = arr_batch
    #         idx += len(arr_batch)
    #     arr.flush()
    #
    # input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    # if not os.path.exists(input_file_path):
    #     data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    #     with open(input_file_path, 'w') as f:
    #         f.write(requests.get(data_url).text)
    #
    # with open(input_file_path, 'r') as f:
    #     data = f.read()
    # print(f"length of dataset in characters: {len(data):,}")
    #
    # # get all the unique characters that occur in this text
    # chars = sorted(list(set(data)))
    # vocab_size = len(chars)
    # print("all the unique characters:", ''.join(chars))
    # print(f"vocab size: {vocab_size:,}")
    #
    # # create a mapping from characters to integers
    # stoi = {ch: i for i, ch in enumerate(chars)}
    # itos = {i: ch for i, ch in enumerate(chars)}
    #
    #
    # def encode(s):
    #     return [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    #
    #
    # def decode(l):
    #     return ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
    #
    #
    # # create the train and test splits
    # n = len(data)
    # train_data = data[:int(n * 0.9)]
    # val_data = data[int(n * 0.9):]
    #
    # # encode both to integers
    # train_ids = encode(train_data)
    # val_ids = encode(val_data)
    # print(f"train has {len(train_ids):,} tokens")
    # print(f"val has {len(val_ids):,} tokens")
    #
    # # export to bin files
    # train_ids = np.array(train_ids, dtype=np.uint16)
    # val_ids = np.array(val_ids, dtype=np.uint16)
    # train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    # val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    #
    # # save the meta information as well, to help us encode/decode later
    # meta = {
    #     'vocab_size': vocab_size,
    #     'itos': itos,
    #     'stoi': stoi,
    # }
    # with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    #     pickle.dump(meta, f)