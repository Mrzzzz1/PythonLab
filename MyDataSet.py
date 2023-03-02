import pickle

from torch.utils.data import Dataset
from Vocabulary import Vocabulary
class MyDataset(Dataset):
    def __init__(self,phrases,labels,sequence_max_len=20):
        '''

        :param phrases: 由单词构成的二维list
        :param labels: 由0，1，2，3，4构成的一维list
        :param sequence_max_len: phrase的最大词数
        '''
        self.phrases = phrases
        self.labels=labels
        self.vocab = pickle.load(open("./models/vocab.pkl", "rb"))
        self.sequence_max_len = sequence_max_len
    def __getitem__(self, index):
        #return self.phrases[index],int(self.labels[index]
        return self.vocab.transform(sentence=self.phrases[index],max_len=self.sequence_max_len), int(self.labels[index])
    def __len__(self):
        return len(self.labels)

    def get_num_embeddings(self):
        return len(self.vocab)

    def get_padding_idx(self):
        return self.vocab.PAD
