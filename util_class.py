import torch
from torch import nn


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dataset, labels):
        'Initialization'
        self.labels = labels
        self.dataset = dataset

  def __len__(self):
        'Denotes the total number of samples'
        return self.dataset.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.dataset[index]
        y = self.labels[index]

        return X, y

  class _GELU(nn.Module):
      """GELU unit approximated by a sigmoid, same as original."""

      def __init__(self):
          super().__init__()

      def forward(self, x: torch.Tensor):
          return torch.sigmoid(1.702 * x) * x
