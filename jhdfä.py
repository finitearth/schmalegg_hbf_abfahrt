from wand.image import Image

import os
from pathlib import PurePath
import glob
# from PIL import Image
from pdf2image import convert_from_path
import pdb


# In[Convert pdf to images]
i = "mckirchy.pdf"
images = convert_from_path(i, dpi=300)

path_split = PurePath(i).parts
fileName, ext = os.path.splitext(path_split[-1])

print(images)
#
#
# f = "mckirchy.pdf"
# with(Image(filename=f, resolution=2)) as source:
#     for i, image in enumerate(source.sequence):
#         # newfilename = f[:-4] + str(i + 1) + '.jpeg'
#         # Image(image).save(filename=newfilename)
#         print(image.get_pixmap())
#         break