from argparse import ArgumentParser
from dataloader import *
from dataset import *
from models import *
from transformers import ViTModel
import torch
from tqdm import tqdm
import pickle
from utils import *
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import ViTFeatureExtractor, ConvNextFeatureExtractor
config = load_config('./config.yaml')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

argument_parser = ArgumentParser()

argument_parser.add_argument(
    '--ckpt_path', type=str, help="Path of the checkpoint", required=True
)
argument_parser.add_argument(
    '--image_path', type=str, help="pkl file to store predictions", required=True
)

arguments = argument_parser.parse_args()
feat_ext_name = config['model']['feat_extractor']
feat_ext_config = config['arch'][feat_ext_name]
if feat_ext_name == 'vit':
    img_preprocessor = ViTFeatureExtractor.from_pretrained(feat_ext_config['args']['pretrained'])
else:
    img_preprocessor = ConvNextFeatureExtractor.from_pretrained(feat_ext_config['args']['pretrained'])
model = model_name_to_class[feat_ext_config['class']](
    pretrained = feat_ext_config['args']['pretrained'], 
    feature_dim = feat_ext_config['args']['feature_dim'] , 
    num_classes = feat_ext_config['args']['num_classes'], 
    dropout_prob = feat_ext_config['args']['dropout_prob'], 
    is_trainable = feat_ext_config['args']['is_trainable']
)

ckpt = torch.load(arguments.ckpt_path)
model.load_state_dict(ckpt["model_state_dict"])
print("\nModel is loaded with the checkpoint")
model.to(device)
image =cv2.imread(arguments.image_path)
image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transformed_image = img_preprocessor(image, return_tensors="pt")["pixel_values"].squeeze(0) # 3, H, 
transformed_image=transformed_image.unsqueeze(0)
transformed_image = transformed_image.to(device)  # Move the input tensor to the same device as the model
with torch.no_grad():
    output = model(transformed_image)
output=output.argmax(dim=1).tolist()
for i in output:
    if i==0:
        print('bacterial_leaf_blight')
    if i==1:
        print('bacterial_leaf_streak')
    if i==2:
        print('bacterial_panicle_blight')
    if i==3:
        print('blast')
    if i==4:
        print('brown_spot')
    if i==5:
        print('dead_heart')
    if i==6:
        print('hispa')
    if i==7:
        print('normal')
    if i==8:
        print('tungro')
