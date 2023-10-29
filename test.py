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
    '--pred_out', type=str, help="pkl file to store predictions", required=True
)

arguments = argument_parser.parse_args()


# The below variable tells what should be used as the feature extractor,
# like vit, or convnext etc. It is specified in config file.
feat_ext_name = config['model']['feat_extractor']
feat_ext_config = config['arch'][feat_ext_name]


# Image preprocessor is sent to AdsNonAds dataset, which will preprocess the input the same way it 
# was done for the images during pretraining of that model.

# NOTE: <something>FeatureExtractor == used for preprocessing purposes. 
# IT WILL NOT RETURN feature vector of an image
img_preprocessor = None

if feat_ext_name == 'vit':
    img_preprocessor = ViTFeatureExtractor.from_pretrained(feat_ext_config['args']['pretrained'])
else:
    img_preprocessor = ConvNextFeatureExtractor.from_pretrained(feat_ext_config['args']['pretrained'])

test_dataset = Paddy(images_dirs=[config['data']['test_dir']],
                        img_preprocessor=img_preprocessor,
                        of_num_imgs=None,
                        overfit_test=False,
                        augment_data=config['data']['augment'])
test_dataloader = make_data_loader(dataset=test_dataset,
                                    batch_size=config['model']['batch_size'],
                                    num_workers=1,
                                    sampler=None,
                                    data_augment=config['data']['augment'])

# We will use model_name_to_class, which is a dictionary that maps name
# of the model, from str to class, to initialize the desired model.
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

pred_labels = []
file_names = []


count_trainable_parameters(model)

criterion = nn.CrossEntropyLoss()

print(len(test_dataset), config['data']['test_dir'])

with torch.no_grad():
    model.eval()
    for batch in tqdm(test_dataloader):
        try:
            inputs, labels, file_name = batch
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            pred_labels.extend(output.argmax(dim=1).tolist())
            file_names.extend(file_name)
        except Exception as e:
            continue

# Normalizing the running loss with dataset length
pred_label=[]
for i in pred_labels:
    if i==0:
        pred_label.append('bacterial_leaf_blight')
    if i==1:
        pred_label.append('bacterial_leaf_streak')
    if i==2:
        pred_label.append('bacterial_panicle_blight')
    if i==3:
        pred_label.append('blast')
    if i==4:
        pred_label.append('brown_spot')
    if i==5:
        pred_label.append('dead_heart')
    if i==6:
        pred_label.append('hispa')
    if i==7:
        pred_label.append('normal')
    if i==8:
        pred_label.append('tungro')
    


final_result = {'feature_extractor': config['model']['feat_extractor'],
                'file_names':file_names,
                'pred_test':pred_label
}
print(pred_labels)
with open(arguments.pred_out, 'wb') as outfile:
    pickle.dump(final_result, outfile)

