from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
import cv2
import torchvision.transforms as T
import os

class Paddy(Dataset):
    def __init__(self,
                 images_dirs,
                 img_preprocessor,
                 of_num_imgs=20,
                 overfit_test=False,
                 augment_data=False,
                 full=0):

        assert type(images_dirs) == list, "Give directory paths in a list, like ['/../dir1'] or ['/../dir1', '/../dir2', ...]"
        
        for directory in images_dirs:
            assert os.path.exists(directory), "The path {} does not exist".format(directory)

        self.data_dirs = images_dirs
        self.augment = augment_data
    
        if overfit_test:
            self.dataset = self.sample_dataset(of_num_imgs)
        else :
            self.dataset = self.train_dataset()
        
        self.image_transforms = img_preprocessor
        print('\n\n', self.image_transforms, '\n')

    def train_dataset(self):
        bacterial_leaf_blight = []
        bacterial_leaf_streak = []
        bacterial_panicle_blight = []
        blast=[]
        brown_spot=[]
        dead_heart=[]
        hispa=[]
        normal=[]
        tungro=[]
        downy_mildew=[]
        for directory in self.data_dirs:
            #might have to change these lines
            bacterial_leaf_blight.extend(glob(directory+'/bacterial_leaf_blight/*'))
            bacterial_leaf_streak.extend(glob(directory+'/bacterial_leaf_streak/*'))
            bacterial_panicle_blight.extend(glob(directory+'/bacterial_panicle_blight/*'))
            blast.extend(glob(directory+'/blast/*'))
            brown_spot.extend(glob(directory+'/brown_spot/*'))
            dead_heart.extend(glob(directory+'/dead_heart/*'))
            hispa.extend(glob(directory+'/hispa/*'))
            normal.extend(glob(directory+'/normal/*'))
            tungro.extend(glob(directory+'/tungro/*'))
            downy_mildew.extend(glob(directory+'/downy_mildew/*'))
        data = []
        data += [[x, 0] for x in bacterial_leaf_blight]
        data += [[x, 1] for x in bacterial_leaf_streak]
        data += [[x, 2] for x in bacterial_panicle_blight]
        data += [[x, 3] for x in blast]
        data += [[x, 4] for x in brown_spot]
        data += [[x, 5] for x in dead_heart]
        data += [[x, 6] for x in hispa]
        data += [[x, 7] for x in normal]
        data += [[x, 8] for x in tungro]
        data+=[[x,9]for x in downy_mildew]
        random.shuffle(data)

        return data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filename = self.dataset[idx][0]
        if os.path.exists(filename) == False:
            print("The file {} does not exist".format(filename))
            return (None,-1)
        # The shape of the image is (height, width, channels)
        # any image is read as BGR image --> converted to RGB
        image =cv2.imread(filename)
        if image is None:
            print("The file {} does not exist".format(filename))
            os.remove(filename)
            return (None,-1)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(cv2.imread(
        #     filename,
        #     flags=cv2.IMREAD_COLOR),
        #     code=cv2.COLOR_BGR2RGB)
        transformed_image = self.image_transforms(image, return_tensors="pt")["pixel_values"].squeeze(0) # 3, H, W
        if self.augment:
            # All the transformation expect the image to be in shape of [...., H, W]
            rotated_90 = T.functional.rotate(transformed_image, angle=90)
            rotated_270 = T.functional.rotate(transformed_image, angle=270) 
            flipped = T.RandomHorizontalFlip(p=1)(transformed_image)

            return_list = [transformed_image, rotated_90, rotated_270, flipped]

            return (return_list, [self.dataset[idx][1]]*len(return_list))
        return (transformed_image, self.dataset[idx][1], filename)
