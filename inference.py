import os
import torch
import argparse
import numpy as np
from srcnn import SRCNN
from PIL import Image
from torchvision.utils import save_image


def transform_image(image):
	print('Opening the image...')
	img = Image.open(image).convert("RGB")
	img = np.array(img, dtype=np.float32)
	img /= 255.
	img = img.transpose([2,0,1])

	return torch.tensor(img, dtype=torch.float)


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--image', default=None, type=str,
						help='path to the image for inference')
	parser.add_argument('-n', '--name', default='result', type=str,
						help='name for the resulting image')
	args = vars(parser.parse_args())
	return args


def main(args):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	os.makedirs('../results', exist_ok=True)

	model = SRCNN().to(device)

	weights = torch.load('../outputs/model_ckpt.pth')
	model.load_state_dict(weights['model_state_dict'])

	model.eval()
	with torch.no_grad():
		image = args['image']
		image_name = args['name']
		image = transform_image(image).to(device)
		print('Processing the image..')
		output = model(image)
		print('Saving the image...')
		save_image(output, f'../results/{image_name}.png')
	print('Done!')

if __name__ == '__main__':
	args = get_args()
	main(args)
	