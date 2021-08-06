import torch
from PIL import Image
from IPython.display import display
import cv2
import matplotlib.pyplot as plt

from fastseg import MobileV3Large
from fastseg.image import colorize, blend

print(torch.__version__)

# model = MobileV3Large.from_pretrained().cuda().eval()
model = MobileV3Large.from_pretrained().eval()

img2 = Image.open('IMG_7282.PNG')
img = cv2.imread('IMG_7282.PNG')
#display(img.resize((800, 400)))
plt.figure("img")
plt.imshow(img2) 

labels = model.predict_one(img)
print('Shape:', labels.shape)
print(labels)

colorized = colorize(labels)
#display(colorized.resize((800, 400)))
plt.figure("colorized")
plt.imshow(colorized) 

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img=Image.fromarray(img)
composited = blend(img, colorized)
#display(composited.resize((800, 400)))
plt.figure("composited")
plt.imshow(composited) 

plt.show()