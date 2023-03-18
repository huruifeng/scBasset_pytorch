import os
import sys
from datetime import datetime
import random

import anndata
import h5py
import numpy as np

from Bio import SeqIO

## read sequences from FASTA file
def read_fasta(fasta_file):
    """
    :param fasta_file:  path to the fasta file
    :return: Dict {chr:seq,}
    """
    if not os.path.exists(fasta_file):
        raise Exception("Error: Fasta file not exist:" + str(fasta_file))

    fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    return fasta_dict


def dna_onehot(seq, seq_len=None, n_uniform=False):
    """
    :param seq: nucleotide sequence.
    :param seq_len: length to extend/trim sequences to.
    :param n_uniform: represent N's as 0.25, forcing float16, rather than sampling.
    :return: seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim: seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2
    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    if n_uniform:
        seq_code = np.zeros((seq_len, 4), dtype="float16")
    else:
        seq_code = np.zeros((seq_len, 4), dtype="bool")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i, 0] = 1
            elif nt == "C":
                seq_code[i, 1] = 1
            elif nt == "G":
                seq_code[i, 2] = 1
            elif nt == "T":
                seq_code[i, 3] = 1
            else:
                if n_uniform:
                    seq_code[i, :] = 0.25
                else:
                    ni = random.randint(0, 3)
                    seq_code[i, ni] = 1
    return seq_code

def make_bed_seqs_from_df(input_bed, fasta_dict, seq_len, stranded=False):
    """Return BED regions as sequences and regions as a list of coordinate
    tuples, extended to a specified length."""
    """Extract and extend BED sequences to seq_len."""


    seqs_dna = []
    seqs_coords = []

    for i in range(input_bed.shape[0]):
        chrm = input_bed.iloc[i,0]
        start = int(input_bed.iloc[i,1])
        end = int(input_bed.iloc[i,2])
        strand = "+"

        # determine sequence limits
        mid = (start + end) // 2
        seq_start = mid - seq_len // 2
        seq_end = seq_start + seq_len

        # save
        if stranded:
            seqs_coords.append((chrm, seq_start, seq_end, strand))
        else:
            seqs_coords.append((chrm, seq_start, seq_end))
        # initialize sequence
        seq_dna = ""
        # add N's for left over reach
        if seq_start < 0:
            print("Adding %d Ns to %s:%d-%s" % (-seq_start, chrm, start, end),file=sys.stderr,)
            seq_dna = "N" * (-seq_start)
            seq_start = 0

        # get dna
        seq_dna += fasta_dict[chrm].seq.upper().__str__()[seq_start:seq_end]

        # add N's for right over reach
        if len(seq_dna) < seq_len:
            print("Adding %d Ns to %s:%d-%s" % (seq_len - len(seq_dna), chrm, start, end),file=sys.stderr,)
            seq_dna += "N" * (seq_len - len(seq_dna))
        # append
        seqs_dna.append(seq_dna)
    return seqs_dna, seqs_coords


def dna_vec(seq, seq_len=None):
    """dna_1hot
    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.
      n_uniform: represent N's as 0.25, forcing float16,
                 rather than sampling.
    Returns:
      seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2
    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    seq_code = np.zeros((seq_len, ), dtype="int8")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i] = 0
            elif nt == "C":
                seq_code[i] = 1
            elif nt == "G":
                seq_code[i] = 2
            elif nt == "T":
                seq_code[i] = 3
            else:
                seq_code[i] =  random.randint(0, 3)
    return seq_code

def split_train_test_val(ids, random_state=10, train_ratio=0.7):
    np.random.seed(random_state)
    test_val_ids = np.random.choice(ids,int(len(ids) * (1 - train_ratio)),replace=False,)
    train_ids = np.setdiff1d(ids, test_val_ids)
    val_ids = np.random.choice(test_val_ids,int(len(test_val_ids) / 2),replace=False,)
    test_ids = np.setdiff1d(test_val_ids, val_ids)
    train_ids.sort()
    test_ids.sort()
    val_ids.sort()
    return train_ids, test_ids, val_ids


def make_h5_sparse(tmp_ad, h5_name, input_fasta, seq_len=1344, batch_size=1000):
    ## batch_size: how many peaks to process at a time
    ## tmp_ad.var must have columns chr, start, end

    t0 = datetime.now()

    n_peaks = tmp_ad.shape[1]
    bed_df = tmp_ad.var.loc[:, ['chr', 'start', 'end']]  # bed file
    bed_df.index = np.arange(bed_df.shape[0])
    n_batch = int(np.floor(n_peaks / batch_size))
    batches = np.array_split(np.arange(n_peaks), n_batch)  # split all peaks to process in batches

    ### create h5 file
    # X is a matrix of n_peaks * 1344
    f = h5py.File(h5_name, "w")

    ds_X = f.create_dataset("X",(n_peaks, seq_len),dtype="int8",)

    # save to h5 file
    for i in range(len(batches)):
        idx = batches[i]
        # write X to h5 file
        seqs_dna, _ = make_bed_seqs_from_df(bed_df.iloc[idx, :],fasta_dict=input_fasta, seq_len=seq_len,)
        dna_array_dense = [dna_vec(x) for x in seqs_dna]
        dna_array_dense = np.array(dna_array_dense)
        ds_X[idx] = dna_array_dense

        t1 = datetime.now()
        total = t1 - t0
        print(f'process {(i+1)*batch_size} peaks takes {total} (Total peaks:{n_peaks})')
    f.close()


