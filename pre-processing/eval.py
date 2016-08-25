classes = ["O", "B-LOC", "B-PER", "B-ORG", "B-TOUR", "I-ORG", "I-PER", "I-TOUR", "I-LOC", "B-PRO", "I-PRO"]


class LongWord:
    def __init__(self):
        self.start = None
        self.stop = None

    def set_start(self, start):
        self.start = start

    def set_stop(self, stop):
        self.stop = stop


def eval(labels, targets):
    print len(labels)
    print len(targets)
    if len(labels) != len(targets):
        raise "Two inputs is not same size"

    long_words = []
    current_word = None
    in_long_word = False
    for i in range(0, len(targets)):
        string_label = classes[targets[i] - 1]

        # meet new long word
        if string_label in ['B-LOC', 'B-PER', 'B-TOUR', 'B-PRO', 'B-ORG', 'O']:
            if in_long_word:
                print "Append {0}".format(targets[current_word.start:current_word.stop + 1])
                long_words.append(current_word)
                in_long_word = False

        # if = 'O'
        if string_label == 'O':
            continue

        if string_label in ['B-LOC', 'B-PER', 'B-TOUR', 'B-PRO', 'B-ORG']:
            current_word = LongWord()
            current_word.set_start(i)
            current_word.set_stop(i)
            in_long_word = True
            continue

        if string_label in ['I-LOC', 'I-PER', 'I-TOUR', 'I-PRO', 'I-ORG']:
            if not in_long_word:
                raise Exception("Fuck at {0}".format(i))

            current_word.set_stop(i)
            continue

    count_true = 0
    for word in long_words:
        if check(labels, targets, word.start, word.stop):
            count_true += 1
    return float(count_true) / len(long_words)


def check(labels, targets, start, stop):
    for i in range(start, stop + 1):
        if labels[i] != targets[i]:
            return False
    return True


# def standard():
#     f1 = open("train_data/full_test_label_file.txt")
#     f2 = open("test_label_standard.txt", "w")
#
#     for line in f1:
#         value = int(line.strip())
#         value += 1
#
#         f2.write("{0}\n".format(str(value)))
#
#     f1.close()
#     f2.close()
#

if __name__ == '__main__':
    f1 = open("test_label_standard.txt")
    f2 = open("result_test.csv")

    labels = []
    targets = []

    for line in f1:
        targets.append(int(line.strip()))

    for line in f2:
        labels.append(int(line.strip()))

    print eval(labels, targets)
