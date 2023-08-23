import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.pie_model import PIENet
import pytorch_lightning as pl



class PIENetInference:
    def __init__(self, ckpt_path):
        # self.model = PIENet(1, 1, [16, 32, 64], [64, 32]).load_from_checkpoint(ckpt_path)
        self.model = PIENet.load_from_checkpoint(ckpt_path, input_channels=1, output_channels=1, encoder_dims=[16, 32, 64], decoder_dims=[64, 32])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4330], std=[0.2349])
        ])

    def __call__(self, input_path, output_path):
        # Read image
        in_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # Pre-processing
        in_img = self.transform(in_img).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            in_img = in_img.to('cuda')
            out_img = self.model(in_img)

        # Post-processing
        out_img = out_img.squeeze().cpu().numpy()
        out_img = np.clip(out_img, 0, 1) * 255
        out_img = out_img.astype(np.uint8)

        # Save output
        cv2.imwrite(output_path, out_img)


def main():
    # Load model
    model = PIENetInference(ckpt_path='lightning_logs/version_0/checkpoints/epoch=27-step=93296.ckpt')

    # Run inference
    model(input_path='/mnt/d/Datasets-hmap/barcode/sample/00008.png', output_path='./out8-27.png')


if __name__ == '__main__':
    # Start main function with experiment name
    main()
