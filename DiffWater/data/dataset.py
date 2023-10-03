from PIL import Image
from torch.utils.data import Dataset
import data.util as Util
from model.ColorChannelCompensation import three_c as t_c

class UIEDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.data_len = data_len
        self.split = split

        self.input_path = Util.get_paths_from_images('{}/input_{}'.format(dataroot, resolution))
        self.target_path = Util.get_paths_from_images('{}/target_{}'.format(dataroot, resolution))

        self.dataset_len = len(self.target_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        target = Image.open(self.target_path[index]).convert("RGB")
        input = Image.open(self.input_path[index]).convert("RGB")
        input = t_c(input)

        [input, target] = Util.transform_augment([input, target], split=self.split, min_max=(-1, 1))

        return {'target': target, 'input': input, 'Index': index}



