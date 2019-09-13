import torch
import torch.nn as nn
from torch.autograd import Variable


class StrLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self):
        self.dict = {}
        # for i, char in enumerate(alphabet):
        #     # NOTE: 0 is reserved for 'blank' required by wrap_ctc
        #     self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        length = []
        result = []
        for item in text:
            item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)

        text = result
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)

            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l])))
                index += l
            return texts

    def convert_integer_to_string(self,integer_sequence):
        total_dictionary = self.encode_total()
        print(integer_sequence)

    def joint_chars(self):
        banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়০১২৩৪৫৬৭৮৯"
        charlist = []  # I have the chars in a list in specific index
        for i in range(0, len(banglachars)):
            charlist.append(banglachars[i])
        str = "ন্ত"
        hosh = str[1]

        char = ""
        jointchars = []

        allowed_for_Po = [31, 38, 41, 39]  # Allowed Character List
        for ind in allowed_for_Po:
            char = charlist[ind] + hosh + "প"
            jointchars.append(char)

        allowed_for_Do = [12, 13, 16, 23, 25, 30, 38, 39]
        for ind in allowed_for_Do:
            char = charlist[ind] + hosh + "ড"
            jointchars.append(char)

        allowed_for_To = [11, 12, 13, 16, 21, 25, 26, 31, 33, 35, 38, 39, 40, 41]
        for ind in allowed_for_To:
            char = charlist[ind] + hosh + "ট"
            jointchars.append(char)

        allowed_for_cho = [16, 18, 20, 21, 24, 25, 31, 38, 39, ]
        for ind in allowed_for_cho:
            char = charlist[ind] + hosh + "চ"
            jointchars.append(char)

        allowed_for_go = [13, 28, 29, 30]
        for ind in allowed_for_go:
            char = charlist[ind] + hosh + "গ"
            jointchars.append(char)

        allowed_for_ko = [11, 15, 38, 40, 41]
        for ind in allowed_for_ko:
            char = charlist[ind] + hosh + "ক"
            jointchars.append(char)

        allowed_for_bo = [11, 12, 13, 14, 17, 18, 21, 23, 25, 26, 27, 28, 29, 30, 33, 35, 38, 39, 40, 41]
        for ind in allowed_for_bo:
            char = charlist[ind] + hosh + "ব"
            jointchars.append(char)

        allowed_for_to = [11, 26, 30, 31, 35, 39]
        for ind in allowed_for_to:
            char = charlist[ind] + hosh + "ত"
            jointchars.append(char)

        allowed_for_do = [13, 28, 30, 33, 35, 38]
        for ind in allowed_for_do:
            char = charlist[ind] + hosh + "দ"
            jointchars.append(char)

        allowed_for_no = [11, 13, 14, 16, 26, 28, 29, 30, 31, 35, 39, 41, 42]
        for ind in allowed_for_no:
            char = charlist[ind] + hosh + "ন"
            jointchars.append(char)

        allowed_for_ro = list(range(11, 46))
        for ind in allowed_for_ro:
            char = charlist[ind] + hosh + "র"
            jointchars.append(char)

        allowed_for_zo = list(range(11, 46))
        for ind in allowed_for_zo:
            char = charlist[ind] + hosh + "য"
            jointchars.append(char)

        allowed_for_bo = list(range(11, 46))
        for ind in allowed_for_bo:
            char = charlist[ind] + hosh + "ব"
            jointchars.append(char)

        char = charlist[11] + hosh + "ষ"
        jointchars.append(char)

        char = charlist[18] + hosh + charlist[18]
        jointchars.append(char)

        char = "ন" + hosh + charlist[18]
        jointchars.append(char)

        char = "ষ" + hosh + "ঠ"
        jointchars.append(char)

        charlist = ["্যা", "ঙ্গ", "প্ল", "ন্স", "ল্ল", "ন্ট", "ন্ধ", "চ্ছ"]
        for i in charlist:
            jointchars.append(i)

        return jointchars

    def get_total_data(self):
        #
        banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়০১২৩৪৫৬৭৮৯"
        charlist = []  # I have the chars in a list in specific index
        for i in range(0, len(banglachars)):
            charlist.append(banglachars[i])

        # chars = "খ্র"  # For generating the "Ri" Character
        # b = (chars[1] + chars[2])  # The "Ri Character consits of two different charcters

        # modifiers = "ঁ া ে ি ী ু ো ৌ ূ ৗ "

        # Add modifiers in the character list
        modifiers = "ঁােিীুোৌূৗ"
        for i in range(0, len(modifiers)):
            charlist.append(modifiers[i])
        joint = self.joint_chars()

        # Total = charlist + modlist + Joint + punc + lowerchar + upperchar
        total = charlist + joint
        return total

    def encode_total(self):
        index = {}
        c = 0
        # string_list =[]
        #
        # for i in range(0, len(string)):
        #     string_list.append(string[i])

        # Encode integer numbers for total dataset
        total = self.get_total_data()
        for i in total:
            index[i] = c
            c += 1
        self.dict = index
        return dict

    def decode_data(self, string):
        index = {}
        c = 0
        # string_list =[]
        #
        # for i in range(0, len(string)):
        #     string_list.append(string[i])

        # Encode integer numbers for total dataset
        total = self.get_total_data()
        for i in total:
            index[i] = c
            c += 1
        i = 0
        labels = []
        while i < len(string):
            isJoint = 0
            isChar = 0
            isNorm = 0

            # for char in total:
            #     if(char == string_list[i]):
            #         labels.append(index[char])
            #         i += 1
            #         break
            #
            # i += 1
            if i + 9 <= len(string):
                ifjointornot = string[i:i + 9]
                flag = 0
                for x in total:
                    if ifjointornot == x:
                        isJoint = 1
                        i += 9
                        labels.append(index[x])
                        break

            if isJoint == 0:
                if i + 3 <= len(string):
                    ifcharornot = string[i:i + 3]
                    flag = 0
                    for x in total:
                        if ifcharornot == x:
                            flag = 1
                            isChar = 1
                            i += 3
                            labels.append(index[x])
                            break
            if isJoint == 0 and isChar == 0:
                for x in total:
                    if x == string[i]:
                        i += 1
                        labels.append(index[x])
                        isNorm = 1
                        break
            if isJoint == 0 and isChar == 0 and isNorm == 0:
                i += 1

        # labels = torch.IntTensor(labels)
        return labels


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    # print(v.size())
    # print(data.size())
    v.data.resize_(data.size()).copy_(data)
    # Img_.resize_(Img.size()).copy_(Img))


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img