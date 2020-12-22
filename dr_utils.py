import bpy
import numpy as np
import os
from mathutils import Vector
import random
import sys
sys.path.append(os.getcwd())

def set_viewport_shading(mode):
    '''Makes color/texture viewable in viewport'''
    areas = bpy.context.workspace.screens[0].areas
    for area in areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = mode

def randomize_light():
    scene = bpy.context.scene
    #scene.view_settings.exposure = random.uniform(2.5,4)
    #scene.view_settings.exposure = random.uniform(2.5,3.7)
    #scene.view_settings.exposure = random.uniform(2,3.7)
    scene.view_settings.exposure = random.uniform(0.6,1.8)
    light_data = bpy.data.lights['Light']
    light_data.color = tuple(np.random.uniform(0,1,3))
    light_data.energy = np.random.uniform(300,600)
    light_data.shadow_color = tuple(np.random.uniform(0,1,3))
    #light_obj = bpy.data.objects['LightObj']
    light_obj = bpy.data.objects['Sun']
    light_obj.data.color = tuple(np.random.uniform(0.8,1,3))
    #light_obj.location = Vector(np.random.uniform(-4,4,3).tolist())
    #light_obj.location[2] = np.random.uniform(4,7)
    light_obj.rotation_euler[0] = np.random.uniform(-np.pi/30, np.pi/30)
    light_obj.rotation_euler[1] = np.random.uniform(-np.pi/30, np.pi/30)
    light_obj.rotation_euler[2] = np.random.uniform(-np.pi/30, np.pi/30)

def randomize_camera():
    scene = bpy.context.scene
    bpy.ops.view3d.camera_to_view_selected()
    dx = np.random.uniform(-0.05,0.05)
    dy = np.random.uniform(-0.05,0.05)
    dz = np.random.uniform(-1,1)
    bpy.context.scene.camera.location += Vector((dx,dy,dz))
    bpy.context.scene.camera.rotation_euler = (0, 0, np.random.uniform(-np.pi/4, np.pi/4))

def pattern(obj, texture_filename):
    '''Add image texture to object (don't create new materials, just overwrite the existing one if there is one)'''
    if '%sTexture' % obj.name in bpy.data.materials: 
        mat = bpy.data.materials['%sTexture'%obj.name]
    else:
        mat = bpy.data.materials.new(name="%sTexture"%obj.name)
        mat.use_nodes = True
    if "Image Texture" in mat.node_tree.nodes:
        texImage = mat.node_tree.nodes["Image Texture"]
    else:
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage.image = bpy.data.images.load(texture_filename)
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    #mat.specular_intensity = np.random.uniform(0, 0.3)
    mat.specular_intensity = np.random.uniform(0, 0)
    mat.roughness = np.random.uniform(0.5, 1)
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

def texture_randomize(obj, textures_folder):
    rand_img_path = random.choice(os.listdir(textures_folder))
    while rand_img_path == '.DS_Store':
        rand_img_path = random.choice(os.listdir(textures_folder))
    img_filepath = os.path.join(textures_folder, rand_img_path)
    pattern(obj, img_filepath)

def color_randomize(obj, color=None):
    noise = (np.random.standard_normal(3))/70.0
    r,g,b = np.array(color) + noise
    color = [r,g,b,1]
    if '%sColor' % obj.name in bpy.data.materials:
        mat = bpy.data.materials['%sColor'%obj.name]
    else:
        mat = bpy.data.materials.new(name="%sColor"%obj.name)
        mat.use_nodes = False
    mat.diffuse_color = color
    mat.specular_intensity = np.random.uniform(0, 0.1)
    mat.roughness = np.random.uniform(0.5, 1)
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
    set_viewport_shading('MATERIAL')
