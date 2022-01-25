import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os


class RuCLIPSBDataset(Dataset):
    def __init__(self, dir, df_path, max_text_len=77):
        self.df = pd.read_csv(df_path)
        self.dir = dir
        self.max_text_len = max_text_len
        self.tokenizer = transformers.BertTokenizer.from_pretrained("cointegrated/rubert-tiny")
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
    def __getitem__(self, idx):
        # достаем имя изображения и ее лейбл
        image_name = self.df['image_name'].iloc[idx]
        text = self.df['text'].iloc[idx]
        input_ids, attention_mask = tokenize(self.tokenizer, [text], max_len=self.max_text_len)
        input_ids, attention_mask = input_ids[0], attention_mask[0]
        image = cv2.imread(os.path.join(self.dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, input_ids, attention_mask
    def __len__(self):
        return len(self.df)
