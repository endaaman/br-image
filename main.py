import os
import re
from glob import glob

from tqdm import tqdm
import pandas as pd
from endaaman.torch import Trainer

from PIL import Image

from datasets import USDataset



class C(Trainer):
    def run_ds(self):
        self.ds = USDataset()

    # def run_train(self):
    #     self.ds = USDataset()

if __name__ == '__main__':
    c = C()
    c.run()
