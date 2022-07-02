import os
import math
import argparse
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, datasets
from model_shufflenet import shufflenet_v2_x1_0
from model_mobilenetv2 import MobileNetV2
from model_resnet import resnet50
from model_densenet import densenet121, load_state_dict
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate, train_one_epoch_Nosoftmax
from torchinfo import summary

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    project_name = args.model_name + '_' + args.data_path.split("/")[-1] + '_' + args.trainset  # --------modified-------- #
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(comment=project_name)

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    if args.model_name == 'ShuffleNet':
        weights = './weights/shufflenetv2_x1-5666bf0f80.pth'
        model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)
    elif args.model_name == 'ResNet':
        weights = './weights/resnet50-19c8e357.pth'
        model = resnet50(num_classes=args.num_classes).to(device)
    elif args.model_name == 'MobileNet':
        weights = './weights/mobilenet_v2-b0353104.pth'
        model = MobileNetV2(num_classes=args.num_classes).to(device)
    elif args.model_name == 'DenseNet':
        weights = './weights/densenet121-a639ec97.pth'
        model = densenet121(num_classes=args.num_classes).to(device)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, args.trainset),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "test"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=args.batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    if weights != "":
        if os.path.exists(weights):
            if args.model_name == 'DenseNet':
                load_state_dict(model, weights)
            else:
                weights_dict = torch.load(weights, map_location='cpu')
                load_weights_dict = {k: v for k, v in weights_dict.items()
                                     if model.state_dict()[k].numel() == v.numel()}
                print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    summary(model, input_size=(1, 3, 224, 224))
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=4E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    print("Mission {} Start!".format(project_name))
    acc_best = 0
    model_last = ''
    for epoch in range(args.epochs):
        # train
        if args.PatchMatch_Aug == True:
            mean_loss = train_one_epoch_Nosoftmax(model=model,
                                        optimizer=optimizer,
                                        data_loader=train_loader,
                                        device=device,
                                        epoch=epoch,
                                        num_classes=args.num_classes,
                                        batch_size=args.batch_size,
                                        PatchMatch_Aug=args.PatchMatch_Aug)
        else:
            mean_loss = train_one_epoch(model=model,
                                        optimizer=optimizer,
                                        data_loader=train_loader,
                                        device=device,
                                        epoch=epoch)

        scheduler.step()

        # validate
        acc = evaluate(model=model,
                       data_loader=validate_loader,
                       device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        if acc > acc_best:
            if os.path.exists(model_last):
                os.remove(model_last)
            torch.save(model.state_dict(), "./weights/{}-{}.pth".format(project_name,epoch))
            acc_best = acc
            model_last = "./weights/{}-{}.pth".format(project_name,epoch)
    print("Mission {} Complete! Best Accurancy: {}".format(project_name,acc_best))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--PatchMatch_Aug', type=bool, default=True) # --------modified-------- default = True
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    # SuffleNetV2_NoSoftmax_Soybean_Small lr = 0.1
    # parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--model_name', type=str, default='ShuffleNet')

    parser.add_argument('--trainset', type=str, default='train_5%_Aug')
    parser.add_argument('--data_path', type=str, default='../data/Soybean')

    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
