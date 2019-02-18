import glfw
from OpenGL.GL import *
import ShaderLoader
import numpy
import pyrr
from PIL import Image
from ObjLoader import *
import random
import math
import os

from image_utils import *

rot_a = 0.0
rot_b = 0.0
rotating = False
pos_x = 0.0
origo_y = -0.25
pos_y = origo_y
translating = False
mouse_pos_x = 0
mouse_pos_y = 0
make_samples_count = 0
eye_distance = 0.5

def key_callback(window, key, scancode, action, mode):
    if key == glfw.KEY_F12 and action == glfw.PRESS:
        global make_samples_count
        make_samples_count = 5000 #20000
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
    obj.load_model("res/Rabbit.obj", scale=0.7)

    texture_offset = len(obj.vertex_index)*12
    normal_offset = (texture_offset + len(obj.texture_index)*8)

    shader = ShaderLoader.compile_shader("shaders/video_18_vert.vs", "shaders/video_18_frag.fs")

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
    image = Image.open("res/Rabbit_D.tga")
    flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = numpy.array(list(flipped_image.getdata()), numpy.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glEnable(GL_TEXTURE_2D)

    glUseProgram(shader)

    glClearColor(0.2, 0.3, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    focus_distance = 1.9
    view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, -focus_distance]))
    projection = pyrr.matrix44.create_perspective_projection_matrix(65.0, w_width / w_height, 0.1, 100.0)
    model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))

    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    model_loc = glGetUniformLocation(shader, "model")
    transform_loc = glGetUniformLocation(shader, "transform")
    light_loc = glGetUniformLocation(shader, "light")

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if make_samples_count > 0:
            rabbit_transform = pyrr.Matrix44.identity(float)
            rabbit_transform[3,0] = pos_x
            rabbit_transform[3,1] = pos_y
            rabbit_transform[3,2] = 0
            
            x_rot = random.random()*2.0*math.pi
            y_rot = random.random()*2.0*math.pi
            z_rot = 0
            euler = pyrr.euler.create(roll=x_rot, pitch=y_rot, yaw=z_rot)
            rot = pyrr.Matrix44.from_eulers(euler)
            #rot *= pyrr.Matrix44.from_eulers(y_rot)
            trans = pyrr.Matrix44.identity(float)
            trans[3,0] = random.random()*1.0 - 0.5
            trans[3,1] = random.random()*0.7 - 0.35
            trans[3,2] = 0 #random.random()*0.2
            trans[3,3] = 1 #1./(0.7 + random.random() * 0.3) # scale

            eye_angle = eye_distance/focus_distance

            q = pyrr.Quaternion.from_matrix(rot)
            
            mat = trans*rot*rabbit_transform
            
            stereo_filename = str(trans[3,0]) + '_' + str(trans[3,1]) + '_' + str(trans[3,2]) + '_' + str(q[0]) + '_' + str(q[1]) + '_' + str(q[2]) + '_' + str(q[3]) + '_' + str(trans[3,3]) + '_' + str(x_rot) + '_' + str(y_rot) + '_' + str(z_rot)

            """
            right_eye_transform = pyrr.Matrix44.from_y_rotation(eye_angle/2)
            left_eye_transform = pyrr.Matrix44.from_y_rotation(-eye_angle/2)
            """
            
            glUniformMatrix4fv(transform_loc, 1, GL_FALSE, mat) #right_eye_transform*mat)
            glUniformMatrix4fv(light_loc, 1, GL_FALSE, mat) #right_eye_transform*mat)
            glDrawArrays(GL_TRIANGLES, 0, len(obj.vertex_index))
            glfw.swap_buffers(window)
            right_array = snapToNumpy()
            
            """
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glUniformMatrix4fv(transform_loc, 1, GL_FALSE, left_eye_transform*mat)
            glUniformMatrix4fv(light_loc, 1, GL_FALSE, left_eye_transform*mat)
            glDrawArrays(GL_TRIANGLES, 0, len(obj.vertex_index))
            glfw.swap_buffers(window)
            left_array = snapToNumpy()
            
            tmp_array = np.stack([left_array,right_array], axis=2).reshape(left_array.shape[0],left_array.shape[1],6)

            array = np.concatenate((tmp_array[:,:,:3],tmp_array[:,:,3:]), axis=1)
            """
            
            array = right_array[:,:,:3]

            if make_samples_count < 201:
                save_to_jpg("./test/"+stereo_filename+'.png', array)
            else:
                save_to_jpg("./train/"+stereo_filename+'.png', array)
            #render_to_jpg(stereo_filename+'_l.png')

            #time.sleep(2.0)
            
            make_samples_count -= 1
           
        else:
            rot = pyrr.Matrix44.from_x_rotation(rot_b)
            rot *= pyrr.Matrix44.from_y_rotation(rot_a)
            trans = pyrr.Matrix44.identity(float)
            trans[3,0] = pos_x
            trans[3,1] = pos_y
            mat = rot*trans
            #print(trans[3,0],trans[3,1],trans[3,2])
    
            glUniformMatrix4fv(transform_loc, 1, GL_FALSE, mat)
            glUniformMatrix4fv(light_loc, 1, GL_FALSE, mat)
            glDrawArrays(GL_TRIANGLES, 0, len(obj.vertex_index))
            glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
