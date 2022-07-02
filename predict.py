import os
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_shufflenet import shufflenet_v2_x1_0
from model_mobilenetv2 import MobileNetV2
from model_resnet import resnet50
from model_densenet import densenet121, load_state_dict
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(args):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    assert os.path.exists(args.img_path), "file: '{}' dose not exist.".format(args.img_path)
    img = Image.open(args.img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # create model
    if args.model_name == 'ShuffleNet':
        model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)
    elif args.model_name == 'ResNet':
        model = resnet50(num_classes=args.num_classes).to(device)
    elif args.model_name == 'MobileNet':
        model = MobileNetV2(num_classes=args.num_classes).to(device)

    weights = './weights/' + args.model_name + '_' + args.plant + '_' + args.trainset + '.pth'
    model.load_state_dict(torch.load(weights, map_location=device))
    # read class_indict
    json_path = './classes/' + args.plant + '.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)


    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        # predict = output
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./images/test2.jpg')
    # parser.add_argument('--img_path', type=str, default='./images/Septoria_test3.jpg')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='MobileNet')
    parser.add_argument('--trainset', type=str, default='train_100%_False')
    parser.add_argument('--plant', type=str, default='Wheat')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
