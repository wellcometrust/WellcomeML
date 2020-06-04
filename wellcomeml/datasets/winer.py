# WiNER: A Wikipedia Annotated Corpus for Named Entity Recognition
# https://www.aclweb.org/anthology/I17-1042/

from wellcomeml.datasets.download import check_cache_and_download

# def load_winer():
import tarfile
import os
from tqdm import tqdm 
import json
import random

def create_train_test(
    NE_path, vocab_path, docs_path,
    train_processed_path, test_processed_path,
    n_sample, prop_train, rand_seed=42
    ):
    """
    NE_path: File path to the named entities
    vocab_path: File path to the vocab dictionary
    docs_path: File path to the documents texts (words coded according to the vocab)
    train_processed_path: Output file path for the training data
    test_processed_path: Output file path for the testing data
    n_sample: Number of files to get data from, each file contains about 1000 documents
    prop_train: Proportion of the file sample that will be in the training data

    Utilise the three datasources to get the entities from a sample
    of documents and save to a training and testing dataset files
    in the form:

    ID 12345
    John Smith PERSON
    rejects O
    German LOC
    call O
    
    Next 0
    sentence 0

    ID 444
    Next
    document
    """

    random.seed(rand_seed)
    n_train = round(prop_train*n_sample)

    print("Creating entities dictionary from {}".format(NE_path))
    entities = {}
    with tarfile.open(NE_path, "r:bz2") as tar:
        for member in tqdm(tar.getmembers()):
            f = tar.extractfile(member)
            if f:
                content = f.read().decode('utf-8', errors='ignore')
                articles = content.replace('\t', ' ').split('ID ')
                if articles[0]=='':
                    articles = articles[1:]
                for article in articles:
                    article_entities = article.split('\n')
                    article_id = article_entities[0]    
                    if article_entities[-1]=='':
                        article_entities = article_entities[1:-2]
                    else:
                        article_entities = article_entities[1:]
                    entity_info = [l.split(' ') for l in article_entities]
                    sentence_entities = {}
                    for sentence_id, begin, end, ent_type in entity_info:
                        sentence_id, begin, end, ent_type = int(sentence_id), int(begin), int(end), int(ent_type)
                        if sentence_entities.get(sentence_id):
                            sentence_entities[sentence_id].append([begin, end, ent_type])
                        else:
                            sentence_entities[sentence_id] = [[begin, end, ent_type]]
                    entities[article_id] = sentence_entities

    # Create id2word from vocab file (line number is id)
    print("Creating id2word dictionary from {}".format(vocab_path))
    id2word = {}
    with open(vocab_path, "r") as vocab:
        f = vocab.read()
        lines = f.split('\n')
        for i, line in enumerate(lines):
            id2word[i] = line.split(' ')[0]

    # Go into a sample of the document files and get the text for these entities
    print("Creating token and tag from {}".format(docs_path))
    with tarfile.open(docs_path, "r:bz2") as tar_docs, \
            open(train_processed_path, 'w') as train_file, \
            open(test_processed_path, 'w') as test_file:
        for i, member in tqdm(enumerate(random.sample(tar_docs.getmembers(), n_sample))):
            if i < n_train:
                output_file = train_file
            else:
                output_file = test_file
            f = tar_docs.extractfile(member)
            if f:
                content = f.read().decode('utf-8', errors='ignore')
                articles = content.replace('\t', ' ').split('ID ')
                if articles[0]=='':
                    articles = articles[1:]
                for article in articles:
                    lines = article.split('\n')
                    article_id = lines[0]
                    article_entities = entities.get(article_id)
                    if article_entities:
                        output_file.write('ID ' + article_id)
                        output_file.write('\n')
                        if lines[-1]=='':
                            lines = lines[1:-1]
                        else:
                            lines = lines[1:]
                        sent_wordidx = [l.split(' ') for l in lines]
                        for sentid, sentence_entities in article_entities.items():
                            if len(sent_wordidx) > int(sentid):
                                tokens = [id2word[int(s)] for s in sent_wordidx[int(sentid)]]
                                tags = ['O']*len(tokens)
                                for begin, end, ent_type in sentence_entities:
                                    tags[(begin+1):(end-1)] = [str(ent_type) + '-I']*(end-1-begin-1)
                                    tags[end-1] = str(ent_type) + '-E'
                                    tags[begin] = str(ent_type) + '-B'
                                for token, tag in list(zip(tokens, tags)):
                                    output_file.write(' '.join([token, tag]))
                                    output_file.write('\n')
                        output_file.write('\n')

def load_data_spacy(data_path, inc_outside=True, merge_entities=True):

    # Load data in Spacy format:
    # X = list of sentences (plural) / documents ['the cat ...', 'some dog...', ...]
    # Y = list of list of entity tags for each sentence [[{'start': 36, 'end': 46, 'label': 'PERSON'}, {..}, ..], ... ]
    # inc_outside = False: don't include none-entities in the output
    # merge_entities if entities span over multiple tags do you want to merge them or not
    # e.g. John 3-B Smith 3-E -> John Smith 3

    X = []
    Y = []
    with open(data_path) as f:
        lines = f.read().split('ID ')
        for line in lines:
            if line != '':
                article_text = ''
                char_i = 0
                article_tags = []
                entities = line.split('\n')
                tokens, tags = zip(*[tuple(ee) for ee in [e.split(' ') for e in entities] if len(ee)==2])
                
                if merge_entities:
                    prev_B = 0
                    group_tags = []
                    for i, tag in enumerate(tags):
                        if tag[-2:]=='-E':
                            group_tags.append((prev_B, i+1))
                        if tag[-2:]=='-B':
                            if len(tags)!=(i+1) and (tags[i+1][-2:]=='-I' or tags[i+1][-2:]=='-E'):
                                prev_B = i
                            else:
                                group_tags.append((i, i+1))
                        if tag=='O':
                            group_tags.append((i, i+1))
                    tags_joined = [' '.join(tags[b:e])[0] for b,e in group_tags]
                    tokens_joined = [' '.join(tokens[b:e]) for b,e in group_tags]
                else:
                    tags_joined = [t[0] for t in tags]
                    tokens_joined = tokens
                for tag, token in zip(tags_joined, tokens_joined):
                    article_text += token + ' '
                    if tag!='O' or inc_outside:
                        article_tags.append({'start': char_i, 'end': char_i+len(token), 'label': tag})
                    char_i += len(token) + 1 # plus 1 for the space separating
                if article_tags!=[]:
                    X.append(article_text)
                    Y.append(article_tags)

    return X, Y

def load_winer(split='train', shuffle=True, inc_outside=True, merge_entities=True):
    path = check_cache_and_download("winer")

    if split == 'train':
        train_data_path = os.path.join(path, "train.txt")
        X, Y = load_data_spacy(train_data_path, inc_outside=inc_outside, merge_entities=merge_entities)
    elif split == 'test':
        test_data_path = os.path.join(path, "test.txt")
        X, Y = load_data_spacy(test_data_path, inc_outside=inc_outside, merge_entities=merge_entities)
    else:
        raise ValueError(f"Split argument {split} is not one of train, test or evaluate")

    if shuffle:
        data = list(zip(X, Y))
        random.shuffle(data)
        X, Y = zip(*data)

    return X, Y

if __name__ == '__main__':

    path = check_cache_and_download("winer")

    train_processed_path = os.path.join(path, "train.txt")
    test_processed_path = os.path.join(path, "test.txt")

    if not os.path.exists(train_processed_path):
        # Since this has been done once it shouldnt need to be done again, 
        # including here for completeness or in th case we want to increase 
        # the sample size
        print("No {} training data file found, generating ...".format(train_processed_path))

        NE_path = os.path.join(path, "CoarseNE.tar.bz2")
        docs_path = os.path.join(path, "Documents.tar.bz2")
        vocab_path = os.path.join(path, "document.vocab")

        # Create the train/test data
        n_sample = 10
        prop_train = 0.7

        create_train_test(
            NE_path, vocab_path, docs_path,
            train_processed_path, test_processed_path,
            n_sample, prop_train, rand_seed=42
            )
    else:
        print("Training and test data found")

