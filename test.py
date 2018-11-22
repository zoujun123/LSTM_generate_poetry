# coding:utf-8
import collections
import numpy as np

MAX_LENGTH = 100
MIN_LENGTH = 10
BEGIN_CHAR = '^'
END_CHAR = '$'
poetry_file = "som-poetry.txt"
max_words = 3000
UNKNOWN_CHAR = "*"

def handle(line):
    if len(line) > MAX_LENGTH:
        index_end = line.rfind('。', 0, MAX_LENGTH)
        index_end = index_end if index_end > 0 else MAX_LENGTH
        line = line[:index_end + 1]
    return BEGIN_CHAR + line + END_CHAR


# 获取了每首诗的正文内容
poetrys = [line.strip().replace(' ', '').split(':')[1] for line in
                open(poetry_file, encoding='utf-8')]

# 添加上开头结尾符号，并通过MAX_LENGTH进行了长度约束
poetrys = [handle(line) for line in poetrys if len(line) > MIN_LENGTH]
# 所有字
words = []
for poetry in poetrys:
    words += [word for word in poetry]
print (words)
counter = collections.Counter(words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
print (count_pairs)
words, _ = zip(*count_pairs)
print (words, _)

words_size = min(max_words, len(words))
words = words[:words_size] + (UNKNOWN_CHAR,)
last_words_size = len(words)

char2id_dict = {w: i for i, w in enumerate(words)}
id2char_dict = {i: w for i, w in enumerate(words)}
unknow_char = char2id_dict.get(UNKNOWN_CHAR)
char2id = lambda char: char2id_dict.get(char, unknow_char)
id2char = lambda num: id2char_dict.get(num)
# 根据长度进行排序
poetrys = sorted(poetrys, key=lambda line: len(line))
# 将每首诗的字转化为id
poetrys_vector = [list(map(char2id, poetry)) for poetry in poetrys]


xdata = np.random.standard_normal([2,3])
print (xdata)
ydata = np.copy(xdata)
print ("\n")
print (ydata)
ydata[:, :-1] = xdata[:, 1:]
print (ydata)
# ydata[:, :-1] = xdata[:, 1:]