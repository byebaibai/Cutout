import cv2
import numpy as np
from model.model import  MobileUnet
import torch
from PIL import Image
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser(description='Testing Setting')
parser.add_argument('--checkpoint', default=None, required=True, help='path to checkpoint')
args = parser.parse_args()
DOWNSIZE = 4

def makeGrid(images):
    row1 = np.hstack([images[0], images[1]])
    row2 = np.hstack([images[2], images[3]])
    image = np.vstack([row1, row2])
    return image

def randomCrop(image, h, w):
    transform = transforms.Compose([transforms.CenterCrop((h, w))])
    Pframe = Image.fromarray(image)
    Pframe = transform(Pframe)
    return np.asarray(Pframe)

img_size = (363, 204)
raw_dir = './raw.avi'
result_dir = './result.avi'
mask_dir = './mask.avi'
back_dir = './back.avi'
fps = 17.151219512195123
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

raw_writer = cv2.VideoWriter(raw_dir, fourcc, fps, img_size)
result_writer = cv2.VideoWriter(result_dir, fourcc, fps, img_size)
mask_writer = cv2.VideoWriter(mask_dir, fourcc, fps, img_size)
back_writer = cv2.VideoWriter(back_dir, fourcc, fps, img_size)


if __name__ == '__main__':
    model = MobileUnet()
    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.eval()
    cap = cv2.VideoCapture(0)
    background = cv2.VideoCapture(0)
    i = 0
    while (True):
        i = i + 1
        ret, frame = cap.read()
        bret, bframe = background.read()
        if(ret == False or bret == False):
            break
        raw = frame
        h, w, _ = frame.shape
        bframe = randomCrop(bframe, h, w)

        bframe = cv2.resize(bframe, (int(frame.shape[1]/DOWNSIZE), int(frame.shape[0]/DOWNSIZE)))
        frame = cv2.resize(frame, (int(frame.shape[1]/DOWNSIZE), int(frame.shape[0]/DOWNSIZE)))

        h, w, _ = frame.shape
        inputs = cv2.resize(raw, (256, 256))
        inputs = (inputs.copy().astype(np.float32) / 255.0).transpose(2, 0, 1)
        inputs = inputs[np.newaxis, :, :, :]
        outputs = model(torch.FloatTensor(inputs))
        outputs = torch.argmax(outputs, dim=1).long()
        outputs = torch.cat([outputs, outputs, outputs], dim=0).float().numpy()
        outputs = outputs.transpose((1, 2, 0))
        outputs = cv2.resize(outputs,(w, h))

        frame = frame/255.0
        bframe = bframe/255.0
        people = frame * outputs
        simulation = bframe * (1 - outputs) + frame * outputs
        cv2.imshow('image', makeGrid([frame, simulation, outputs, people]))

        raw_writer.write(np.uint8(frame * 255))
        result_writer.write(np.uint8(people * 255))
        mask_writer.write(np.uint8(outputs * 255))
        back_writer.write(np.uint8(simulation * 255))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    raw_writer.release()
    result_writer.release()
    mask_writer.release()
    back_writer.release()
    cv2.destroyAllWindows()