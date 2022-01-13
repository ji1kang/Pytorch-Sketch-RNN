import torch
import random
import numpy as np
from logging_helper import Logger
logger_instance = Logger('dataloader')
logger = logger_instance.logger


# preprocessing for storkes


def get_bounds(data, factor=10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


def slerp(p0, p1, t):
    """Spherical interpolation."""
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def lerp(p0, p1, t):
    """Linear interpolation."""
    return (1.0 - t) * p0 + t * p1


def pad_image(img, max_len):
    """Pad the batch to be stroke-5 bigger format as described in paper."""
    result = np.zeros((1, max_len + 1, 5), dtype=float)
    l = len(img)

    #assert l <= max_len
    result[0, 0:l, 0:2] = img[0:l, 0:2]
    result[0, 0:l, 3] = img[0:l, 2]
    result[0, 0:l, 2] = 1 - result[0, 0:l, 3]
    result[0, l:, 4] = 1
    # put in the first token, as described in sketch-rnn methodology
    result[0, 1:, :] = result[0, :-1, :]
    result[0, 0, :] = 0
    result[0, 0, 2] = 1  # setting S_0 from paper.
    result[0, 0, 3] = 0
    result[0, 0, 4] = 0
    return result

# A note on formats:
# Sketches are encoded as a sequence of strokes. stroke-3 and stroke-5 are
# different stroke encodings.
#   stroke-3 uses 3-tuples, consisting of x-offset, y-offset, and a binary
#       variable which is 1 if the pen is lifted between this position and
#       the next, and 0 otherwise.
#   stroke-5 consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
#   one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.
#   See section 3.1 of https://arxiv.org/abs/1704.03477 for more detail.
# Sketch-RNN takes input in stroke-5 format, with sketches padded to a common
# maximum length and prefixed by the special start token [0, 0, 1, 0, 0]
# The QuickDraw dataset is stored using stroke-3.


def strokes_to_lines(strokes):
    """Convert stroke-3 format to polyline format."""
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
            lines.append(line)
            line = []
        else:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
    return lines


def lines_to_strokes(lines):
    """Convert polyline format to stroke-3 format."""
    eos = 0
    strokes = [[0, 0, 0]]
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :]


def augment_strokes(strokes, prob=0.0):
    """Perform data augmentation by randomly dropping out strokes."""
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            result.append(stroke)
    return np.array(result)


def scale_bound(stroke, average_dimension=10.0):
    """Scale an entire image to be less than a certain size."""
    # stroke is a numpy array of [dx, dy, pstate], average_dimension is a float.
    # modifies stroke directly.
    bounds = get_bounds(stroke, 1)
    max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    stroke[:, 0:2] /= (max_dimension / average_dimension)


def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result


def clean_strokes(sample_strokes, factor=100):
    """Cut irrelevant end points, scale to pixel space and store as integer."""
    # Useful function for exporting data to .json format.
    copy_stroke = []
    added_final = False
    for j in range(len(sample_strokes)):
        finish_flag = int(sample_strokes[j][4])
        if finish_flag == 0:
            copy_stroke.append([
                int(round(sample_strokes[j][0] * factor)),
                int(round(sample_strokes[j][1] * factor)),
                int(sample_strokes[j][2]),
                int(sample_strokes[j][3]), finish_flag
            ])
        else:
            copy_stroke.append([0, 0, 0, 0, 1])
            added_final = True
            break
    if not added_final:
        copy_stroke.append([0, 0, 0, 0, 1])
    return copy_stroke


def to_big_strokes(stroke, max_len=250):
    """Converts from stroke-3 to stroke-5 format and pads to given length."""
    # (But does not insert special start token).

    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result


def get_max_len(strokes):
    """Return the maximum length of an array of strokes."""
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len


# DataLoader

class DataLoader(torch.utils.data.Dataset):  # Map-style datasets
    def __init__(self,
                 strokes, labels,
                 batch_size=100,
                 max_seq_length=250,
                 normalize_scale_factor=1.000003, scale_factor=1.0,
                 random_scale_factor=0.0,
                 augment_stroke_prob=0.0,
                 limit=1000, sampler=False, shuffle=True):

        if sampler and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')
        self.sampler = sampler
        self.shuffle = shuffle
        self.labels = labels
        self.batch_size = batch_size  # minibatch size
        self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
        self.normalize_scale_factor = normalize_scale_factor  # scale_factor from trainset
        self.scale_factor = scale_factor  # divide offsets by this factor
        self.random_scale_factor = random_scale_factor  # data augmentation method
        # Removes large gaps in the data. x and y offsets are clamped to have
        # absolute value no greater than this limit.
        self.limit = limit
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
        self.start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
        # sets self.strokes (list of ndarrays, one per sketch, in stroke-3 format,
        # sorted by size)
        self.preprocess(strokes)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.strokes)

    def preprocess(self, strokes):
        """Remove entries from strokes having > max_seq_length points."""
        raw_data = []
        seq_len = []
        count_data = 0
        new_labels = []
        for i in range(len(strokes)):
            data = strokes[i]
            if len(data) <= (self.max_seq_length):
                count_data += 1
                # removes large gaps from the data
                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data, dtype=np.float32)
                data[:, 0:2] /= self.scale_factor
                raw_data.append(data)
                seq_len.append(len(data))
                new_labels.append(self.labels[i])
        seq_len = np.array(seq_len)  # get strokes' length for each sketch
        logger.info(f"total images <= max_seq_len is {count_data}")
        self.num_batches = int(count_data / self.batch_size)

        self.strokes = []
        self.labels = []

        if self.sampler:
            idx = np.argsort(seq_len)  # sort sketches by length
            for i in range(len(seq_len)):
                self.strokes.append(raw_data[idx[i]])
                self.labels.append(new_labels[idx[i]])
        elif self.shuffle:
            idx = np.random.permutation((np.arange(len(seq_len))))
            for i in range(len(seq_len)):
                self.strokes.append(raw_data[idx[i]])
                self.labels.append(new_labels[idx[i]])
        else:
            self.strokes = raw_data
            self.labels = new_labels

    def random_sample(self):
        """Return a random sample, in stroke-3 format as used by draw_strokes."""
        sample = np.copy(random.choice(self.strokes))
        return sample

    def random_scale(self, data):
        """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
        x_scale_factor = (
            np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (
            np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def calculate_normalizing_scale_factor(self):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(self.strokes)):
            if len(self.strokes[i]) > self.max_seq_length:
                continue
            for j in range(len(self.strokes[i])):
                data.append(self.strokes[i][j, 0])
                data.append(self.strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)

    def normalize(self, normalizing_scale_factor=None):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        for i in range(len(self.strokes)):
            self.strokes[i][:, 0:2] /= self.normalize_scale_factor

    def _get_batch_from_indices(self, indices):
        """Given a list of indices, return the potentially augmented batch."""
        x_batch = []  # stroke-3 format: List(np.array)
        seq_len = []  # list of seq_len: List(int)
        x_labels = []  # list of x_label: List(int)
        for idx in range(len(indices)):
            i = indices[idx]
            data = self.random_scale(self.strokes[i])
            data_copy = np.copy(data)
            if self.augment_stroke_prob > 0:
                data_copy = augment_strokes(
                    data_copy, self.augment_stroke_prob)
            x_batch.append(data_copy)
            length = len(data_copy)
            seq_len.append(length)
            x_labels.append(self.labels[i])
        seq_len = np.array(seq_len, dtype=int)
        # We return four things: stroke-5 format, list of seq_len, and list of x_labels.
        # convert as tensor
        #pad_5_strokes = torch.from_numpy(self.pad_batch(x_batch, self.max_seq_length))
        #seq_len = torch.tensor(seq_len, dtype=int)
        #x_labels = torch.tensor(x_labels)
        return x_batch, self.pad_batch(x_batch, self.max_seq_length), seq_len, x_labels

    def random_batch(self):
        """Return a randomised portion of the training data."""
        idx = np.random.permutation(range(0, len(self.strokes)))[
            0:self.batch_size]
        return self._get_batch_from_indices(idx)

    def __getitem__(self, idx):
        """Get the idx'th batch from the dataset."""
        assert idx >= 0, "idx must be non negative"
        assert idx <= self.num_batches, "idx must be less than the number of batches"
        start_idx = idx * self.batch_size
        indices = range(start_idx, start_idx + self.batch_size)
        return self._get_batch_from_indices(indices)

    def pad_batch(self, batch, max_len):
        """Pad the batch to be stroke-5 bigger format as described in paper."""
        result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
        assert len(batch) == self.batch_size
        for i in range(self.batch_size):
            l = len(batch[i])
            assert l <= max_len
            result[i, 0:l, 0:2] = batch[i][:, 0:2]
            result[i, 0:l, 3] = batch[i][:, 2]
            result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
            result[i, l:, 4] = 1
            # put in the first token, as described in sketch-rnn methodology
            result[i, 1:, :] = result[i, :-1, :]
            result[i, 0, :] = 0
            # setting S_0 from paper.
            result[i, 0, 2] = self.start_stroke_token[2]
            result[i, 0, 3] = self.start_stroke_token[3]
            result[i, 0, 4] = self.start_stroke_token[4]
        return result  # shape: (batch_size, max_seq_len, 5)


# Load dataloaders (train, test, vaild)

def copy_hparams(class_instance):
    from copy import deepcopy
    return deepcopy(class_instance)


def load_dataset(model_params):
    """Loads the .npz file, and splits the set into train/valid/test."""
    import os

    # normalizes the x and y columns using the training set.
    # applies same scaling factor to valid and test set.

    if isinstance(model_params.data_set, list):
        datasets = model_params.data_set
    else:
        datasets = [model_params.data_set]

    train_strokes = None
    valid_strokes = None
    test_strokes = None

    # for classification
    train_y = None
    valid_y = None
    test_y = None

    for idx, dataset in enumerate(datasets):
        data_filepath = os.path.join(model_params.data_dir, dataset)
        data = np.load(data_filepath, encoding='latin1', allow_pickle=True)
        logger.info('Loaded {}/{}/{} from {}'.format(
            len(data['train']), len(data['valid']), len(data['test']),
            dataset))

        if train_strokes is None:
            train_strokes = data['train']
            valid_strokes = data['valid']
            test_strokes = data['test']
            train_y = [idx]*len(train_strokes)
            valid_y = [idx]*len(valid_strokes)
            test_y = [idx]*len(test_strokes)
        else:
            train_strokes = np.concatenate((train_strokes, data['train']))
            valid_strokes = np.concatenate((valid_strokes, data['valid']))
            test_strokes = np.concatenate((test_strokes, data['test']))
            train_y = np.concatenate((train_y, [idx]*len(data['train'])))
            valid_y = np.concatenate((valid_y, [idx]*len(data['valid'])))
            test_y = np.concatenate((test_y, [idx]*len(data['test'])))

    all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
    num_points = 0
    for stroke in all_strokes:
        num_points += len(stroke)
    avg_len = num_points / len(all_strokes)
    logger.info(
        f"""Dataset combined: {len(all_strokes)} ({len(train_strokes)}/{len(valid_strokes)}/{len(test_strokes)}), avg len {int(avg_len)}""")

    # calculate the max strokes we need.
    max_seq_len = get_max_len(all_strokes)
    # overwrite the hps with this calculation.
    model_params.max_seq_len = max_seq_len

    logger.info('model_params.max_seq_len %i.', model_params.max_seq_len)

    eval_model_params = copy_hparams(model_params)

    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 0

    sample_model_params = copy_hparams(eval_model_params)
    sample_model_params.batch_size = 1  # only sample one at a time

    train_set = DataLoader(
        strokes=train_strokes, labels=train_y,
        batch_size=model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor,
        augment_stroke_prob=model_params.augment_stroke_prob,
        sampler=False, shuffle=True)

    # Todo:
    normalizing_scale_factor = model_params.normalizing_scale_factor
    #model_params.normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
    train_set.normalize()

    valid_set = DataLoader(
        strokes=valid_strokes, labels=valid_y,
        batch_size=eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0,
        sampler=False, shuffle=True)
    valid_set.normalize()

    test_set = DataLoader(
        strokes=test_strokes, labels=test_y,
        batch_size=eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0,
        sampler=True, shuffle=False)
    test_set.normalize()

    result = [
        train_set, valid_set, test_set,
        model_params, eval_model_params, sample_model_params
    ]

    return result

# for pred


def processed_strokes(strokes, hp):
    """
    - storkes: sketch data generated by the Canvas
    return: 3-format-stroke

    Useage:
    # flower_strokes = [[[313,311,309,307,304,300,297,294,292,291,291,292,295,297,301,306,312,317,323,329,336,342,347,353,358,364,370,374,379,382,385,387,389,389,390,390,389,387,384,381,378,373,369,364,360,355,351,346,341,337,332,328,322,317,314,311,309,308,306,306],[240,241,243,246,250,255,262,269,277,285,293,299,307,314,319,325,329,333,336,338,339,339,338,336,333,329,324,319,314,307,301,296,291,285,278,272,265,261,257,253,250,246,242,240,238,236,235,234,234,234,234,234,235,236,237,238,239,241,243,244],[1540,1574,1587,1605,1620,1638,1655,1672,1689,1705,1722,1739,1755,1772,1789,1805,1822,1839,1856,1872,1890,1905,1922,1939,1956,1972,1989,2005,2022,2039,2056,2071,2089,2105,2122,2138,2155,2173,2189,2205,2222,2238,2255,2271,2288,2305,2322,2341,2356,2375,2389,2405,2422,2439,2455,2472,2488,2508,2525,2530]],[[287,285,283,278,273,267,261,254,246,240,233,227,222,217,212,207,203,201,199,199,200,202,205,208,212,217,222,227,233,240,247,254,260,266,271,278,287,295,303,310,316,320,324,330,335,338,342,345,347,348,349,350,351,351,351,351,351,351,350,349,348,347,346,346,345,345],[271,271,271,271,269,268,266,264,262,258,254,250,245,240,235,229,224,218,211,204,196,189,183,176,169,162,157,152,148,142,137,134,131,129,127,126,124,123,123,123,125,127,131,136,143,148,154,161,169,176,182,188,194,199,205,210,214,218,222,225,228,231,233,234,234,233],[2853,2865,2871,2885,2903,2919,2938,2955,2971,2988,3004,3021,3038,3055,3071,3089,3105,3121,3138,3155,3171,3188,3205,3221,3239,3255,3271,3287,3304,3320,3337,3355,3372,3391,3406,3425,3438,3455,3470,3487,3504,3521,3538,3556,3571,3588,3604,3620,3637,3653,3671,3689,3706,3723,3738,3754,3770,3787,3803,3820,3837,3855,3872,3887,3904,3969]],[[348,348,348,349,350,352,353,356,358,360,362,365,368,371,374,378,383,389,395,402,407,413,418,423,429,434,440,445,451,456,460,465,469,473,477,480,482,485,488,489,490,490,490,489,487,484,481,476,471,466,461,456,452,448,444,439,435,431,427,422,417,412,408,404,401,398,396,394,393,392],[231,230,228,224,218,212,207,202,195,189,183,178,173,169,166,164,161,158,154,152,150,149,148,148,149,151,153,155,157,160,163,166,170,175,179,184,187,194,199,205,210,214,219,224,228,231,235,239,244,247,251,254,257,260,262,264,266,267,269,271,273,275,277,277,278,278,278,278,278,278],[4236,4270,4287,4303,4320,4338,4354,4370,4388,4403,4422,4437,4459,4472,4489,4505,4521,4537,4554,4576,4591,4606,4621,4637,4656,4670,4687,4704,4721,4737,4757,4772,4787,4804,4819,4837,4852,4874,4887,4905,4922,4937,4954,4970,4987,5003,5020,5037,5053,5069,5087,5103,5120,5136,5153,5170,5187,5204,5222,5237,5257,5270,5285,5302,5319,5337,5354,5370,5387,5403]],[[395,396,397,400,403,408,414,419,424,429,434,440,445,450,456,460,465,470,474,477,480,483,485,486,487,489,489,489,489,487,485,482,478,473,468,462,457,452,446,440,433,427,421,415,410,403,399,394,390,386,382,379,375,373,370,367,365,363,361,359,357,355,355,354,353,353,352,352],[279,279,278,278,278,279,280,282,283,284,286,289,292,295,299,304,309,314,319,323,328,333,337,341,343,347,351,356,361,367,372,378,383,388,394,398,401,403,404,405,405,405,405,405,405,403,402,399,396,392,388,384,380,377,373,370,366,363,360,357,354,352,351,350,349,348,346,345],[5673,5702,5719,5735,5752,5770,5787,5806,5820,5836,5852,5870,5886,5903,5920,5936,5955,5971,5988,6002,6019,6035,6052,6073,6075,6086,6104,6121,6139,6153,6171,6186,6202,6219,6235,6254,6271,6286,6305,6319,6336,6352,6369,6386,6408,6421,6436,6452,6469,6485,6502,6519,6541,6552,6570,6587,6602,6620,6636,6655,6669,6686,6703,6722,6737,6752,6773,6786]],[[354,354,355,357,358,358,358,358,356,354,350,346,342,338,334,330,325,320,316,310,302,295,287,281,275,269,263,258,252,247,242,237,232,227,222,218,214,210,209,205,201,199,197,196,195,195,195,196,198,200,202,205,208,210,213,216,220,225,230,235,240,245,251,256,261,267,270,271,276,280,282,284,285,286],[352,354,358,364,371,378,385,393,400,405,410,415,419,422,425,427,430,433,435,438,441,444,447,448,449,449,448,446,444,441,439,436,433,430,426,423,419,415,414,408,403,397,390,384,379,374,369,363,357,351,345,340,334,329,324,318,313,307,302,297,293,289,285,282,279,277,275,275,274,273,273,273,273,273],[7052,7068,7082,7100,7116,7134,7151,7168,7185,7201,7218,7235,7251,7268,7284,7301,7318,7335,7351,7368,7385,7401,7419,7435,7452,7469,7485,7501,7519,7534,7551,7568,7585,7602,7619,7635,7651,7668,7669,7685,7702,7718,7735,7751,7769,7785,7802,7818,7835,7851,7869,7884,7901,7918,7937,7951,7968,7984,8002,8018,8035,8051,8070,8086,8102,8118,8135,8136,8151,8167,8190,8203,8218,8252]]]
    # flower_strokes_filted, seq_len = processed_strokes(flower_strokes,  hp.normalizing_scale_factor)
    # import vizualization as vis
    # viz.draw_strokes(flower_strokes_filted, factor=.88)
    """
    _3_strokes = []
    init_points = strokes[0][0][:2]  # (x, y)
    max_seq_len = 0

    for points in strokes:
        max_seq_len += len(points[0])

    # concat strokes as an array
    for i, points in enumerate(strokes):
        x, y, time = points

        num_points = len(x)

        if max_seq_len >= hp.max_seq_len:
            resize = num_points // 2
            processed = np.zeros((resize, 3))
            processed[:, 0] = [x[i] for i in range(1, num_points, 2)]
            processed[:, 1] = [y[i] for i in range(1, num_points, 2)]
        else:
            processed = np.zeros((num_points, 3))
            processed[:, 0] = [x[i] for i in range(num_points)]
            processed[:, 1] = [y[i] for i in range(num_points)]

        processed[-1, 2] = 1  # end_stroke
        _3_strokes.append(processed)
    _3_strokes = np.concatenate(_3_strokes)

    # normalize
    _3_strokes[:, 0] -= init_points[0]
    _3_strokes[:, 1] -= init_points[1]
    _3_strokes[1:, 0] = _3_strokes[1:, 0] - _3_strokes[:-1, 0]
    _3_strokes[1:, 1] = _3_strokes[1:, 1] - _3_strokes[:-1, 1]

    #_3_strokes[:, :2] /= hp.normalizing_scale_factor

    return _3_strokes
