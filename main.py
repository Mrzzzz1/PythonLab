import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
import gzip
import csv
import time
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Vocabulary import Vocabulary
from MyDataSet import MyDataset
from MyModule import  MyModule
import nlpaug.augmenter.word as naw
from nlpaug.flow import Sometimes
import matplotlib.pyplot as plt

BATCH_SIZE=128

def data_enhance(sentence,num):
    stop_words = []
    synonym_aug = naw.SynonymAug(stopwords=stop_words)
    spelling_aug = naw.SpellingAug(stopwords=stop_words, aug_p=0)
    # 将多种数据增强方式融合
    aug = Sometimes([synonym_aug, spelling_aug])
    r = aug.augment(sentence, num)
    return r


def tokenlize(sentence):
    """
    进行文本分词
    :param sentence: str
    :return: [str,str,str]
    """

    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“','it','is','are','of','for']
    sentence = sentence.lower()  # 把大写转化为小写
    sentence = re.sub("<br />", " ", sentence)
    # sentence = re.sub("I'm","I am",sentence)
    # sentence = re.sub("isn't","is not",sentence)
    sentence = re.sub("|".join(fileters), " ", sentence)
    result = [i for i in sentence.split(" ") if len(i) > 0]

    return result


def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews, labels = zip(*batch)

    return reviews, labels

def build_vocabulary(phrases_word_by_word):
    '''
    建立词汇表模型并存到"./models/vocab.pkl"
    :param phrases_word_by_word: [['it', "'s", 'certainly', 'an', 'honest', 'attempt', 'to', 'get', 'at', 'something']]
    :return:
    '''
    vocab = Vocabulary()
    for phrase in tqdm(phrases_word_by_word, total=len(phrases_train_word_by_word)):
        vocab.fit(phrase)

    vocab.build_vocab()
    print(len(vocab))
    if not os.path.exists("./models"):
        os.makedirs("./models")
    pickle.dump(vocab, open("./models/vocab.pkl", "wb"))

def read_data():
    '''
    读取数据分为训练数据和测试数据并返回
    :return: phrases_train_word_by_word，phrases_val_word_by_word:由word构成的二维list [[str]]
            phrases_val,labels_val：由label组成的一维list [str]
    '''
    with open('train.tsv') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            rows = list(reader)
        data = [[row[2],row[3]] for row in rows]
    data_train, data_val = train_test_split(data, test_size = 0.1)
    phrases_train = []#[data[0] for data in data_train]
    labels_train = []#[data[1] for data in data_train]
    now=0
    for data in data_train:
        phrases_train.append(data[0])
        labels_train.append(data[1])
    #     r = data_enhance(data[0],1)
    #     phrases_train.append(r[0])
    #     labels_train.append(data[1])
        # if data[1]=='0' or data[1]=='4':
        #     r = data_enhance(data[0],2)
        #     phrases_train.append(r[0])
        #     labels_train.append(data[1])
        #     phrases_train.append(r[1])
        #     labels_train.append(data[1])


        # now+=1
        # if(now%100==0):
        #     print(now)

    # flag = True
    # for data in data_train:
    #     if data[1]=='2':
    #         if flag == True:
    #             phrases_train.append(data[0])
    #             labels_train.append(data[1])
    #             flag = False
    #         else : flag = True
    #     else :
    #         phrases_train.append(data[0])
    #         labels_train.append(data[1])

    # for data in data_train:
    #     if data[1]=='0' or data[1] == '4' :
    #         phrases_train.append(data[0])
    #         labels_train.append(data[1])
    phrases_train_word_by_word=[]

    for phrase in phrases_train:
        phrase_word_by_word = tokenlize(phrase)
        phrases_train_word_by_word.append(phrase_word_by_word)

    phrases_val = [data[0] for data in data_val]
    labels_val = [data[1] for data in data_val]
    phrases_val_word_by_word=[]
    dataNum=[0,0,0,0,0]
    for data in data_train:
        dataNum[int(data[1])]+=1
    print(dataNum)

    for phrase in phrases_val:
        phrase_word_by_word = tokenlize(phrase)
        phrases_val_word_by_word.append(phrase_word_by_word)

    return phrases_train_word_by_word,labels_train,phrases_val_word_by_word,labels_val

def trainModel():
    correct=0
    total_loss = 0
    for i, (inputs,target) in enumerate(train_loader, 1):
        input = torch.stack(inputs,0)
        #print(target)

        output = classifier(input)
        pred = output.max(dim=1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        #print(output.size())
        #print(target.size())
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(loss)
            print(f'[{i * len(inputs[0])}/{len(dataset_train)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs[0]))}')
            print(correct)
    return total_loss

def testModel():
    correct = 0
    total = len(dataset_val)
    print("evaluating trained model ...")
    with torch.no_grad():
        testNum=[0,0,0,0,0]
        testTar = [0,0,0,0,0]
        for i, (inputs,target) in enumerate(val_loader, 1):
            inputs = torch.stack(inputs, 0)
            output = classifier(inputs)
            pred = output.max(dim=1, keepdim=True)[1]
            for i in pred:
                testNum[pred[0]]+=1
            for i in target:
                testTar[i]+=1
            correct += pred.eq(target.view_as(pred)).sum().item()
        percent = '%.2f' % (100 * correct / total)
        print(testNum)
        print(testTar)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')

    return correct / total


if __name__ == '__main__':
    phrases_train_word_by_word,labels_train,phrases_val_word_by_word,labels_val = read_data()
    build_vocabulary(phrases_train_word_by_word)
    dataset_train = MyDataset(phrases_train_word_by_word,labels_train,sequence_max_len=20)
    dataset_val = MyDataset(phrases_val_word_by_word,labels_val,sequence_max_len=20)
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    classifier = MyModule(dataset_train.get_num_embeddings(),256,5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    acc_list = [0]
    for epoch in range(0,3):
        trainModel()
        acc = testModel()
        print(acc)
        acc_list.append(acc)
        epochs = [epoch for epoch in range(len(acc_list))]
        import matplotlib.pyplot as plt
        plt.figure(0)
        plt.plot(epochs, acc_list)
        x_major_locator = plt.MultipleLocator(1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.savefig("./acc.png")
        plt.show()

