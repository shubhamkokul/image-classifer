from PIL import Image
import numpy as np

img = Image.open('/home/castiel/Artificial Intelligence/classifier/Airplanes-Cars (1)/Airplanes-Cars/20 Cars/20 Cars 20x20/image002.jpg').convert('L')

np_img = np.array(img)
np_img = ~np_img  # invert B&W
np_img[np_img > 0] = 1

print(np_img)



