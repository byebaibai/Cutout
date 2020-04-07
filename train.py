import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_ssim
from dataset import FastDataset
from model import *
from utils import *
from visualization import *
import argparse

parser = argparse.ArgumentParser(description='Training Setting')
parser.add_argument('--train_image_dir', required=True, help='training image path')
parser.add_argument('--train_label_dir', required=True, help='training label path')
parser.add_argument('--valid_image_dir', required=True, help='validation image path')
parser.add_argument('--valid_label_dir', required=True, help='validation label path')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--checkpoint', default=None, help='path to checkpoint')
parser.add_argument('--lr', default=6e-4, type=float, help='learning rate')
parser.add_argument('--epoch', default=200, type=int, help='epoch number')
parser.add_argument('--log_freq', default=100, type=int, help='log frequency')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

MEAN = [0.4560,  0.4472,  0.4155]
count = 0
batch_time = AverageMeter('batch_time')
data_time = AverageMeter('data_time')
losses_D = AverageMeter('losses_D')
losses_B = AverageMeter('losses_B')
losses_V = AverageMeter('losses_V')
losses_S = AverageMeter('losses_S')

dataset_train = FastDataset(args.train_image_dir, args.train_label_dir)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
dataset_test = FastDataset(args.valid_image_dir, args.valid_label_dir)
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False)


model = MobileUnet()
if args.checkpoint is not None:
    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
DiceCriterion = BinaryDiceLoss()
SsimCriterion = pytorch_ssim.SSIM(window_size=11,size_average=True)
print('# parameters:', sum(param.numel() for param in model.parameters()))


def validation(epoch):
    model.eval()
    losses_V.reset()
    val_start_time = time.time()
    raw = None
    label = None
    pred = None
    for i, data in enumerate(dataloader_test):
        inputs, labels = data['image'], data['label']
        inputs = toVariable(inputs.type(torch.FloatTensor))
        labels = toVariable(labels.type(torch.FloatTensor))
        labels_one_hot = toVariable(torch.nn.functional.one_hot(labels.to(torch.int64), 2).transpose(1, 4).squeeze(-1))

        outputs = model(inputs)
        gt = torch.argmax(outputs, dim=1).long()

        raw = inputs
        label = labels
        pred = gt

        loss_B = F.cross_entropy(outputs, gt)
        loss_Dice = DiceCriterion(outputs, labels_one_hot)
        loss_SSIM = 1 - SsimCriterion(outputs, labels_one_hot.float())
        loss = loss_B + loss_Dice + loss_SSIM
        losses_V.update(float(loss.cpu().data.numpy()))

    pred = pred[:, np.newaxis, :, :]
    val_time = time.time() - val_start_time
    torch.save(model.state_dict(), 'model_{}.pkl'.format(epoch))
    print("------EPOCH: {}, VAL_TIME: {:.3f}, VAL_LOSS: {:.4f}------".format(epoch, val_time, losses_V.avg))
    vis = torch.cat(postProcess(raw, pred, label), dim=2)
    validShow(losses_V.avg, epoch, vis)

def train(epoch):
    global count
    model.train()
    end = time.time()
    for i, data in enumerate(dataloader_train):
        inputs, labels = data['image'], data['label']
        inputs = toVariable(inputs.type(torch.FloatTensor))
        labels = toVariable(labels.type(torch.FloatTensor))
        labels_one_hot = toVariable(torch.nn.functional.one_hot(labels.to(torch.int64), 2).transpose(1, 4).squeeze(-1))

        optimizer.zero_grad()

        outputs = model(inputs)
        gt = torch.argmax(outputs,dim=1).long().to(device)
        loss_B = F.cross_entropy(outputs, gt)
        loss_Dice = DiceCriterion(outputs, labels_one_hot)
        loss_SSIM = 1 - SsimCriterion(outputs, labels_one_hot.float())
        loss = loss_B + loss_Dice + loss_SSIM
        losses_B.update(float(loss_B.cpu().data.numpy()))
        losses_D.update(float(loss_Dice.cpu().data.numpy()))
        losses_S.update(float(loss_SSIM.cpu().data.numpy()))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_freq == 0:
            count = count + 1
            text = 'Epoch: [{0}][{1}/{2}] Time {batch_time.val:.3f} ({batch_time.avg:.3f}) ' \
                   'Dice: {loss_D.val:.4f}({loss_D.avg:.4f})  Focal: {loss_B.val:.4f}({loss_B.avg:.4f})  ' \
                   'SSIM: {loss_S.val:.4f}({loss_S.avg:.4f}) ' .format(
                   epoch, i, len(dataloader_train), batch_time=batch_time,
                loss_D=losses_D, loss_B=losses_B, loss_S = losses_S)
            print(text)
            Visualization(gen_loss=losses_B.avg,disc_loss=losses_D.avg,
                          ssim_loss=losses_S.avg ,Count=count,text=text,image=None)

if __name__ == '__main__':
    validation(0)
    for i in range(1, args.epoch + 1):
        train(i)
        validation(i)