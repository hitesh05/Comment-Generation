import os
import re
import csv
import time
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass
import torch
import random
import numpy as np
import argparse
# from metric import Metric
from torch.utils.data import SequentialSampler, BatchSampler
from datasets import load_dataset, Dataset

class Option:
    dataset_label: str  # basts gh sit so
    
class DataLoader:
    @staticmethod
    def _load_basts_data(lang, split):
        # assert lang in ('java', 'python')
        assert split in ('train', 'valid', 'test')
        # data_dir = Path.cwd().parent / 'data' / 'basts' / lang
        data_dir = Path.cwd().parent / 'adamo' / 'data' / 'basts' / 'java'
        with open(data_dir / split / f'{split}.token.code') as file:
            sources = [line.strip().lower() for line in file]
        with open(data_dir / split / f'{split}.token.nl') as file:
            targets = [line.strip().lower() for line in file]
        return Dataset.from_dict({'snippets': sources, 'comments': targets})

    @staticmethod
    def _load_sit_data(lang, split):
        assert lang in ('java', 'python')
        assert split in ('train', 'valid', 'test')
        data_dir = Path.cwd().parent / 'data' / 'sit' / lang
        with open(data_dir / f'{split}.token.code') as file:
            sources = [line.strip().lower() for line in file]
        with open(data_dir / f'{split}.token.nl') as file:
            targets = [line.strip().lower() for line in file]
        return Dataset.from_dict({'snippets': sources, 'comments': targets})

    @staticmethod
    def _load_so_data(lang, split):
        assert lang in ('java', 'python')
        assert split in ('train', 'valid')
        split = 'val' if split == 'valid' else split
        # split: train / val
        data_dir = Path.cwd().parent / 'data' / 'so' / 'pair' / lang
        with open(data_dir / f'{split}.src') as file:
            sources = [line.strip().lower() for line in file]
        with open(data_dir / f'{split}.tgt') as file:
            targets = [line.strip().lower() for line in file]
        return Dataset.from_dict({'snippets': sources, 'comments': targets})

    @staticmethod
    def load_data(label, lang, split, dryrun=False):
        assert label in ('basts')
        assert lang in ('java')
        assert split in ('train', 'valid', 'test')
        # split: train / valid / test
        # dataset: basts sit gh so
        if label == 'basts':
            dataset = DataLoader._load_basts_data(lang, split)
            print("Basts data loaded")
        # elif label == 'sit':
        #     dataset = DataLoader._load_sit_data(lang, split)
        # elif label == 'gh':
        #     dataset = DataLoader._load_gh_data(lang, split)
        # elif label == 'so':
        #     dataset = DataLoader._load_so_data(lang, split)
        else:
            raise NotImplementedError
        # print(dataset['snippets'][:3])
        # print(dataset['comments'][:3])
        if dryrun:
            dataset = dataset.select(range(800)) if split == 'train' else dataset.select(range(100))
        print(dataset[69])
        return dataset

    @staticmethod
    def clean_dataset(dataset):
        def _clean_code(code_data):
            code_tokens = code_data.split()
            code_tokens = list(filter(lambda x: x.isalnum(), code_tokens))
            return ' '.join(code_tokens)

        def _clean_text(text_data):
            text_tokens = text_data.split()
            text_tokens = list(filter(lambda x: x.isalnum(), text_tokens))
            return ' '.join(text_tokens)

        cleaned_dataset = dataset.map(
            lambda sample: {
                'snippets': _clean_code(sample['snippets']),
                'comments': _clean_text(sample['comments']),
            }
        )
        return cleaned_dataset

    @staticmethod
    def refine_dataset(dataset, task):
        assert task in ('ca', 'ce', 'ci')

        def _refine(src, tgt):
            src_tokens = src.split()
            tgt_tokens = tgt.split()
            common_tokens = set(src_tokens) & set(tgt_tokens)
            if task == 'ca':
                return ' '.join(['@+@' if token in common_tokens else '@-@' for token in tgt_tokens])
            elif task == 'ce':
                return ' '.join(['@+@' if token in common_tokens else token for token in tgt_tokens])
            elif task == 'ci':
                return ' '.join(['@-@' if token not in common_tokens else token for token in tgt_tokens])
            else:
                raise NotImplementedError

        refined_dataset = dataset.map(
            lambda sample: {
                'snippets': sample['snippets'],
                'comments': _refine(sample['snippets'], sample['comments']),
            }
        )
        return refined_dataset
    
obj = DataLoader()
obj.load_data('basts', 'java','train')