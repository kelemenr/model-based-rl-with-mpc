from torch.utils.data import Dataset


class TransitionData(Dataset):
    def __init__(self, dataset):
        """
        Init.
        """
        self.dataset = dataset

    def __getitem__(self, item):
        """
        Returns a transition from the dataset.
        :param item: index of the data to be returned.
        :return: dataset[i]
        """
        return self.dataset[item]

    def __len__(self):
        """
        Gives the length size of the dataset.
        :return: len(dataset)
        """
        return len(self.dataset)
