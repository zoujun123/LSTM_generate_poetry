# coding:utf-8
import collections

MAX_LENGTH = 100
MIN_LENGTH = 10
BEGIN_CHAR = '^'
END_CHAR = '$'
poetry_file = "poetry.txt"
max_words = 3000
UNKNOWN_CHAR = "*"

def batch_yield(x_train, batch_size):

    curr_x = []
    for line in x_train:

        if len(curr_x) == batch_size:
            yield curr_x
            curr_x = []
        curr_x.append(line)

    if len(curr_x) != 0:
        yield curr_x

def handle(line):
    if len(line) > MAX_LENGTH:
        index_end = line.rfind('。', 0, MAX_LENGTH)
        index_end = index_end if index_end > 0 else MAX_LENGTH
        line = line[:index_end + 1]
    return BEGIN_CHAR + line + END_CHAR

def read_poetrys():
    poetrys = [line.strip().replace(' ', '').split(':')[1] for line in open(poetry_file, encoding='utf-8')]
    poetrys = [handle(line) for line in poetrys if len(line) > MIN_LENGTH]
    return poetrys


def construct_dictionary(poetrys):
    # poetrys = [line.strip().replace(' ', '').split(':')[1] for line in open(poetry_file, encoding='utf-8')]
    # poetrys = [handle(line) for line in poetrys if len(line) > MIN_LENGTH]
    words = []
    for poetry in poetrys:
        words += [word for word in poetry]
    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    # [('，', 50), ('。', 50), ('^', 9), ('$', 9),...
    words, _ = zip(*count_pairs)
    # ('，', '。', '^', '$',...)  (50, 50, 9, 9,...)

    words_size = min(max_words, len(words))
    words = words[:words_size] + (UNKNOWN_CHAR,)
    # 字典容量：min(max_words, len(words)) + 1
    words_size = len(words)
    char2id_dict = {w: i for i, w in enumerate(words)}
    id2char_dict = {i: w for i, w in enumerate(words)}
    return char2id_dict, id2char_dict

def char2id(char, char2id_dict):
    unknow_char = char2id_dict.get(UNKNOWN_CHAR)
    return char2id_dict.get(char, unknow_char)

def id2char(num, id2char_dict):
    return id2char_dict.get(num)


def construct_poetry_idvector():
    poetrys = read_poetrys()
    char2id_dict, id2char_dict = construct_dictionary(poetrys)
    words_size = len(char2id_dict)
    # 根据长度进行排序
    poetrys = sorted(poetrys, key=lambda line: len(line))
    # 将每首诗的字转化为id
    poetrys_vectors = []
    for poetry in poetrys:

        temp = []
        for char in poetry:
            num = char2id(char, char2id_dict)
            temp.append(num)
        poetrys_vectors.append(temp)
    return poetrys_vectors, words_size, char2id_dict, id2char_dict


if __name__ == '__main__':
    poetry_vectors, words_size, char2id_dict = construct_poetry_idvector()
    print (char2id_dict.get("*"))