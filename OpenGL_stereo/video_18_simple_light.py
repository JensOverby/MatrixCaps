import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import pyrr
from PIL import Image
import os
import random
import math
import time


class ObjLoader:
    def __init__(self):
        self.vert_coords = []
        self.text_coords = []
        self.norm_coords = []

        self.vertex_index = []
        self.texture_index = []
        self.normal_index = []

        self.model = []

    def load_model(self, file, scale=1.0):
        for line in open(file, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue

            if values[0] == 'v':
                mylist = []
                for value in values[1:4]:
                    mylist.append(str(float(value)*scale))
                self.vert_coords.append(mylist)
                #self.vert_coords.append(values[1:4])
            if values[0] == 'vt':
                self.text_coords.append(values[1:3])
            if values[0] == 'vn':
                self.norm_coords.append(values[1:4])

            if values[0] == 'f':
                face_i = []
                text_i = []
                norm_i = []
                for v in values[1:4]:
                    w = v.split('/')
                    face_i.append(int(w[0])-1)
                    text_i.append(int(w[1])-1)
                    norm_i.append(int(w[2])-1)
                self.vertex_index.append(face_i)
                self.texture_index.append(text_i)
                self.normal_index.append(norm_i)

        self.vertex_index = [y for x in self.vertex_index for y in x]
        self.texture_index = [y for x in self.texture_index for y in x]
        self.normal_index = [y for x in self.normal_index for y in x]

        for i in self.vertex_index:
            self.model.extend(self.vert_coords[i])

        for i in self.texture_index:
            self.model.extend(self.text_coords[i])

        for i in self.normal_index:
            self.model.extend(self.norm_coords[i])

        self.model = np.array(self.model, dtype='float32')

def render_to_jpg(filename, format="PNG"):
    #filename = "dump_1.png"
    os.chdir(r"./dumps")
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

    x, y, width, height = glGetDoublev(GL_VIEWPORT)
    width, height = int(width), int(height)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image = image.resize((28,28), Image.ANTIALIAS)
    image.save(filename, format)
    os.chdir(r"..")

rot_a = 0.0
rot_b = 0.0
rotating = False
pos_x = 0.0
origo_y = -0.4
pos_y = origo_y
translating = False
mouse_pos_x = 0
mouse_pos_y = 0
make_samples_count = 0
eye_distance = 0.2

def key_callback(window, key, scancode, action, mode):
    if key == glfw.KEY_F12 and action == glfw.PRESS:
        global make_samples_count
        make_samples_count = 1000
        #render_to_jpg()
    if key == glfw.KEY_ESCAPE:
        global rot_a
        global rot_b
        global pos_x
        global pos_y
        rot_a = 0.0
        rot_b = 0.0
        pos_x = 0.0
        pos_y = origo_y

def cursor_pos_callback(window, x, y):
    global mouse_pos_x
    global mouse_pos_y
    if rotating:
        global rot_a
        global rot_b
        rot_a += (mouse_pos_x-x)/100.0
        rot_b += (mouse_pos_y-y)/100.0
    elif translating:
        global pos_x
        global pos_y
        pos_x += -(mouse_pos_x-x)/200.0
        pos_y += (mouse_pos_y-y)/200.0
    mouse_pos_x = x
    mouse_pos_y = y

def mouse_button_callback(window, button, action, mods):
    if button == glfw.MOUSE_BUTTON_LEFT:
        global rotating
        rotating = (action == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        global translating
        translating = (action == glfw.PRESS)
        if not translating:
            print ("pos_x=",pos_x)
            print ("pos_y=",pos_y)

def scroll_callback(window, xOffset, yOffset):
    print (xOffset, yOffset)

def load_shader(shader_file):
    shader_source = ""
    with open(shader_file) as f:
        shader_source = f.read()
    f.close()
    return str.encode(shader_source)

def compile_shader(vs, fs):
    vert_shader = load_shader(vs)
    frag_shader = load_shader(fs)

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vert_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(frag_shader, GL_FRAGMENT_SHADER))
    return shader

def window_resize(window, width, height):
    minimum = min(width,height)
    glViewport(0, 0, minimum, minimum)

def main():

    global make_samples_count

    # initialize glfw
    if not glfw.init():
        return

    w_width, w_height = 800, 800

    #glfw.window_hint(glfw.RESIZABLE, GL_FALSE)

    window = glfw.create_window(w_width, w_height, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, window_resize)
    glfw.set_key_callback(window, key_callback)
    
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    obj = ObjLoader()
    obj.load_model("Rabbit.obj", scale=0.7)

    texture_offset = len(obj.vertex_index)*12
    normal_offset = (texture_offset + len(obj.texture_index)*8)

    shader = compile_shader("video_18_vert.c", "video_18_frag.c")

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, obj.model.itemsize * len(obj.model), obj.model, GL_STATIC_DRAW)

    #positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, obj.model.itemsize * 3, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    #textures
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, obj.model.itemsize * 2, ctypes.c_void_p(texture_offset))
    glEnableVertexAttribArray(1)
    #normals
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, obj.model.itemsize * 3, ctypes.c_void_p(normal_offset))
    glEnableVertexAttribArray(2)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # load image
    image = Image.open("Rabbit_D.tga")
    flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.array(list(flipped_image.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glEnable(GL_TEXTURE_2D)

    glUseProgram(shader)

    glClearColor(0.2, 0.3, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    focus_distance = 1.0
    view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, -focus_distance]))
    projection = pyrr.matrix44.create_perspective_projection_matrix(65.0, w_width / w_height, 0.1, 100.0)
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))

    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    model_loc = glGetUniformLocation(shader, "model")
    transform_loc = glGetUniformLocation(shader, "transform")
    #light_loc = glGetUniformLocation(shader, "light")

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if make_samples_count > 0:
            rot = pyrr.Matrix44.from_x_rotation(random.random()*2*math.pi)
            rot *= pyrr.Matrix44.from_y_rotation(random.random()*2*math.pi)
            trans = pyrr.Matrix44.identity(float)
            trans[3,0] = pos_x #+ random.random()*0.4 - 0.2
            trans[3,1] = pos_y #+ random.random()*0.2 - 0.1
            eye_angle = eye_distance/focus_distance

            q = pyrr.Quaternion.from_matrix(rot)
            stereo_filename = '0.0_0.0_0.0_' + str(q[0]) + '_' + str(q[1]) + '_' + str(q[2]) + '_' + str(q[3])

            right_eye_transform = pyrr.Matrix44.from_y_rotation(eye_angle/2)
            left_eye_transform = pyrr.Matrix44.from_y_rotation(-eye_angle/2)
            
            glUniformMatrix4fv(transform_loc, 1, GL_FALSE, right_eye_transform*rot*trans)
            #glUniformMatrix4fv(light_loc, 1, GL_FALSE, rot*trans)
            glDrawArrays(GL_TRIANGLES, 0, len(obj.vertex_index))
            glfw.swap_buffers(window)
            render_to_jpg(stereo_filename+'_r.png')

            time.sleep(0.1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glUniformMatrix4fv(transform_loc, 1, GL_FALSE, left_eye_transform*rot*trans)
            #glUniformMatrix4fv(light_loc, 1, GL_FALSE, rot*trans)
            glDrawArrays(GL_TRIANGLES, 0, len(obj.vertex_index))
            glfw.swap_buffers(window)
            render_to_jpg(stereo_filename+'_l.png')

            time.sleep(0.1)
            
            make_samples_count -= 1
           
        else:
            rot = pyrr.Matrix44.from_x_rotation(rot_b)
            rot *= pyrr.Matrix44.from_y_rotation(rot_a)
            trans = pyrr.Matrix44.identity(float)
            trans[3,0] = pos_x
            trans[3,1] = pos_y
    
            glUniformMatrix4fv(transform_loc, 1, GL_FALSE, rot*trans)
            #glUniformMatrix4fv(light_loc, 1, GL_FALSE, rot*trans)
            glDrawArrays(GL_TRIANGLES, 0, len(obj.vertex_index))
            glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()