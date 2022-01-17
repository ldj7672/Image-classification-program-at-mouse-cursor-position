import pyautogui
import keyboard
from PIL import ImageFont
from PIL import ImageDraw

from torchvision import transforms
from torchvision.models import mobilenet_v2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def make_class_idx(file_name):
    file = open(file_name,"r")
    str = file.readlines()
    class_idx_label = {}
    for i in range(len(str)):
        a = int(str[i].split(':')[0])
        b = str[i].split(':')[1]
        class_idx_label[a] = b
    file.close
    return class_idx_label

def image_classify(category):
    if category =='obj' or category =='all':
        obj_idx_label = make_class_idx('class_idx.txt')
        model_obj = mobilenet_v2(pretrained=True)
        model_obj.eval()
        model_obj.cuda()
    if category =='mat' or category =='all':
        mat_idx_label = make_class_idx('minc_idx.txt')
        model_mat = models.resnet50()
        model_mat = torch.nn.DataParallel(model_mat)
        model_mat = torch.load('resnet50_minc_pretrained.pt')
        model_mat.eval()
        model_mat.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    print('========이미지 분류 준비 완료========')
    print('마우스 커서 위치 캡처 : p, 종료 : q')

    while True:
        if keyboard.read_key() == "p":
            y,x=pyautogui.position()
            imgHalfSize=112

            x_top = x-imgHalfSize
            y_left = y-imgHalfSize
            img = pyautogui.screenshot(region=(y_left,x_top,224,224))
            import copy
            img_view = copy.copy(img)
            
            img = transform(img).cuda()
            img = img.reshape(1,3,224,224)
            font = ImageFont.truetype("malgun.ttf", 15)
            draw = ImageDraw.Draw(img_view)

            if category =='obj' or category =='all':
                output_obj = model_obj(img)
                prob_obj = F.softmax(output_obj, dim=1)
                prob_obj = str(prob_obj.data.max(1)[0].item()*100)
                prob_obj = prob_obj[0:5]+'%'
                pred_num_obj = output_obj.data.max(1)[1].item()
                label_obj = obj_idx_label[pred_num_obj]
                label_obj = label_obj+'('+prob_obj+')'
                draw.text((3, 1), label_obj, font=font, fill="blue")

            if category =='mat' or category =='all':
                output_mat = model_mat(img)
                prob_mat = F.softmax(output_mat, dim=1)
                prob_mat = str(prob_mat.data.max(1)[0].item()*100)
                prob_mat = prob_mat[0:5]+'%'
                pred_num_mat = output_mat.data.max(1)[1].item()
                label_mat = mat_idx_label[pred_num_mat]
                label_mat = label_mat+'('+prob_mat+')'
                draw.text((3, 37), label_mat, font=font, fill="red")
            img_view.show()
        if keyboard.read_key()=="q":
            break

if __name__ == '__main__':
    image_classify('all') 
    # all : object + material
    # obj : object
    # mat : material