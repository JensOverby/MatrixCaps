'''
Created on Jan 14, 2019

@author: jens
'''
import glfw
from OpenGL.GL import *
from PIL import Image
import numpy as np
import os

def snapToNumpy():
    x, y, width, height = glGetDoublev(GL_VIEWPORT)
    width, height = int(width), int(height)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image = image.resize((100,100), Image.ANTIALIAS)
    array = np.array(image)
    #im = Image.fromarray(array)
    return array

def save_to_jpg(filename, array, format="PNG"):
    #filename = "dump_1.png"
    #os.chdir(r"./dumps")
    while True:
        file = ''
        for file in os.listdir(os.curdir):
            if file == filename:
                name, ext = file.split(".")
                word, number = name.split("%")
                new_num = int(number) + 1
                filename = word + "%" + str(new_num) + "." + ext
                file = filename
                break
            else:
                continue
        if (file != filename):
            break

    image = Image.fromarray(array)
    image.save(filename, format)
    #os.chdir(r"..")
