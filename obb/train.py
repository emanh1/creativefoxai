import torch
from ultralytics import YOLO

def main():

    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.is_available())
    model = YOLO("best.pt")  
    model.train(data='data.yaml', epochs=50, device=0)

if __name__ == '__main__':
    main()