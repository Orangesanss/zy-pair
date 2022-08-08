'''
Biaffine Dependency parser from AllenNLP
'''
import argparse
import json
import os
import re
import sys
from nltk import word_tokenize
from allennlp.predictors.predictor import Predictor
from lxml import etree
import nltk
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm

MODELS_DIR = '../data'
model_path = os.path.join(
    MODELS_DIR, "biaffine-dependency-parser-ptb-2020.04.06.tar.gz")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--model_path', type=str, default=model_path,
                        help='Path to biaffine dependency parser.')
    parser.add_argument('--data_path', type=str, default='../data/semeval14_pair',
                        help='Directory of where semeval14 or twiiter data held.')
    return parser.parse_args()




def xml2txt(file_path):
    '''
    Read the original xml file of semeval data and extract the text that have aspect terms.
    Store them in txt file.
    '''
    output = file_path.replace('.xml', '_text_pair.txt')
    sent_list = []
    with open(file_path, 'rb') as f:
        raw = f.read()
        root = etree.fromstring(raw)
        for sentence in root:
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            if terms:
                for apect in terms:
                    tar = apect.attrib['term']
                    sent = sentence.find('text').text + 'How do you think of the %s?' % (tar)
                    sent_list.append(sent)
    with open(output, 'w') as f:
        for s in sent_list:
            f.write(s + '\n')
    print('processed', len(sent_list), 'of', file_path)


def text2docs(file_path, predictor):
    '''
    Annotate the sentences from extracted txt file using AllenNLP's predictor.
    '''
    with open(file_path, 'r') as f:
        sentences = f.readlines()
    docs = []
    print('Predicting dependency information...')
    for i in tqdm(range(len(sentences))):
        docs.append(predictor.predict(sentence=sentences[i]))

    return docs


def dependencies2format(doc):  # doc.sentences[i]
    '''
    Format annotation: sentence of keys
                                - tokens
                                - tags
                                - predicted_dependencies
                                - predicted_heads
                                - dependencies
    '''
    sentence = {}
    sentence['tokens'] = doc['words']
    # print(sentence['tokens'])
    sentence['tags'] = doc['pos']
    # sentence['energy'] = doc['energy']
    predicted_dependencies = doc['predicted_dependencies']
    predicted_heads = doc['predicted_heads']
    sentence['predicted_dependencies'] = doc['predicted_dependencies']
    sentence['predicted_heads'] = doc['predicted_heads']
    sentence['dependencies'] = []
    for idx, item in enumerate(predicted_dependencies):
        dep_tag = item
        frm = predicted_heads[idx]
        to = idx + 1
        sentence['dependencies'].append([dep_tag, frm, to])

    return sentence


def get_dependencies(file_path, predictor):
    docs = text2docs(file_path, predictor)
    sentences = [dependencies2format(doc) for doc in docs]
    return sentences


def syntaxInfo2json(sentences, origin_file):

    json_data = []
    tk = TreebankWordTokenizer()
    mismatch_counter = 0
    idx1 = 0
    with open(origin_file, 'rb') as fopen:
        raw = fopen.read()
        root = etree.fromstring(raw)
        for sentence in root:
            # for RAN
            terms = sentence.find('aspectTerms')
            senten_text = sentence.find('text').text
            if terms is None:
                # idx1 += 1
                continue

            # example['energy'] = sentences[idx]['energy']
            for c in terms:
                print('-----------------------------------------------------------------')
                target = c.attrib['term']
                if c.attrib['polarity'] == 'conflict':
                    idx1 += 1
                    continue
                example = dict()
                example['tokens'] = sentences[idx1]['tokens']
                example['tags'] = sentences[idx1]['tags']
                example['predicted_dependencies'] = sentences[idx1]['predicted_dependencies']
                example['predicted_heads'] = sentences[idx1]['predicted_heads']
                example['dependencies'] = sentences[idx1]['dependencies']
                example["sentence"] = senten_text+'How do you think of the %s?'%(target)
                example["aspect_sentiment"] = []
                example['from_to'] = []  # left and right offset of the target word
                example["aspect_sentiment"].append((target, c.attrib['polarity']))
                aspect = [i for i in str(target).split(' ')]
                # We would tokenize the aspect while at it.
                # aspect = word_tokenize(aspect)
                print('tokens: ',example['tokens'])
                print('target: ', target)
                frm =[]
                for (i,j) in enumerate(example['tokens']):
                    if '-' in aspect[0]:
                        s = aspect[0].split('-')[0]
                        if j == s:
                            frm.append(i)
                    elif 'plastic' in aspect[0]:
                        if j == 'plastic':
                            frm.append(i)
                    elif '/' in aspect[0]:
                        s = aspect[0].split('/')[0]
                        if '"' in s:
                            if s[1:] == j :
                                frm.append(i-1)
                        if j == s and '"' not in s:
                            frm.append(i)
                    elif 'uninstall' in aspect[0]:
                        if j == 'uninstall':
                            frm.append(i)
                    elif 'GB' in aspect[0]:
                        if j == 'GB':
                            frm.append(i-1)
                    elif 'gb' in aspect[0]:
                        if j == 'gb':
                            frm.append(i-1)
                    elif '16Gb' in aspect[0]:
                        if j == '16Gb':
                            frm.append(i)
                    elif '4G' in aspect[0]:
                        if j == '4':
                            frm.append(i)
                    elif '3G' in aspect[0]:
                        if j == '3':
                            frm.append(i)
                    elif '8G' in aspect[0]:
                        if j == '8':
                            frm.append(i)
                    elif '"tools"' in aspect[0]:
                        if j == 'tools':
                            frm.append(i-1)
                    elif '"WLAN"' in aspect[0]:
                        if j == 'WLAN':
                            frm.append(i-1)
                    elif '"sales"' in aspect[0]:
                        if j == 'sales':
                            frm.append(i-1)
                    elif '21"' in aspect[0]:
                        if j == '21':
                            frm.append(i)
                    elif '22"' in aspect[0]:
                        if j == '22':
                            frm.append(i)
                    elif '15"' in aspect[0]:
                        if j == '15':
                            frm.append(i)
                    elif '17"' in aspect[0]:
                        if j == '17':
                            frm.append(i)
                    elif '30"' in aspect[0]:
                        if j == '30':
                            frm.append(i)
                    elif "Dell's" in aspect[0]:
                        if j == 'Dell':
                            frm.append(i)
                    elif "PC's" in aspect[0]:
                        if j == 'PC':
                            frm.append(i)
                    elif "window's" in aspect[0]:
                        if j == 'window':
                            frm.append(i)
                    else:
                        if j == aspect[0]:
                            frm.append(i)
                to =len(example["tokens"])-1
                print(frm[-1], to)
                print('-- aspect: ',example["tokens"][frm[-1]:to])
                # print('-----------------------------------------')
                example['from_to'].append((frm[-1], to))
                json_data.append(example)
                idx1 += 1

    extended_filename = origin_file.replace('.xml', '_biaffine_depparsed_pair.json')
    with open(extended_filename, 'w') as f:
        json.dump(json_data, f)
    print('done', len(json_data))
    # print(idx)


def main():
    args = parse_args()

    predictor = Predictor.from_path(args.model_path)

    # data = [('Restaurants_Train_v2.xml', 'Restaurants_Test_Gold.xml'),
    #         ('Laptop_Train_v2.xml', 'Laptops_Test_Gold.xml')]

    data = [
        ('Laptop_Train_v2.xml', 'Laptops_Test_Gold.xml')]
    for train_file, test_file in data:
    # for train_file in data:
        # xml -> txt
        xml2txt(os.path.join(args.data_path, train_file))
        xml2txt(os.path.join(args.data_path, test_file))

        # txt -> json
        train_sentences = get_dependencies(
            os.path.join(args.data_path, train_file.replace('.xml', '_text_pair.txt')), predictor)
        test_sentences = get_dependencies(os.path.join(
            args.data_path, test_file.replace('.xml', '_text_pair.txt')), predictor)

        print(len(train_sentences), len(test_sentences))
        # print(len(train_sentences))

        syntaxInfo2json(train_sentences, os.path.join(args.data_path, train_file))
        syntaxInfo2json(test_sentences, os.path.join(args.data_path, test_file))
if __name__ == "__main__":
    main()
