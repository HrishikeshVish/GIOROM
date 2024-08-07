import sys, os
#sys.path.append(os.path.abspath(os.getcwd())) # change this to your path to “path/to/BlenderToolbox/
import blendertoolbox as bt
#import BlenderToolBox as bt
import bpy, bmesh
import numpy as np
cwd = os.getcwd()
owl_id = 5
#path = f'Owl_blender/giorom_owl_{owl_id}/'
path = f'plasticine_blender/plasticine_{owl_id}/'
if(os.path.exists(os.path.join(cwd, path))==False):
    os.mkdir(os.path.join(cwd, path))
for idx in range(320):
    
    #outputPath = f'/home/csuser/Documents/Neural Operator/Results/Owl_blender/owl_{owl_id}/'+ str(idx+1) + '.png'
    outputPath = path + str(idx+1) + '.png'

    ## initialize blender
    imgRes_x = 1480 
    imgRes_y = 1480 
    numSamples = 10 
    exposure = 1.5 
    bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

    ## read mesh (choose either readPLY or readOBJ)
    #meshPath = f'/home/csuser/Documents/Neural Operator/Results/owl_Licrom_res/owl{owl_id}/pred/pred_'+ str(idx+1) + '.obj'
    meshPath = f'/home/csuser/Documents/Neural Operator/Results/plasticine_res/plasticine_{owl_id}/'+ str(idx+1) + '.obj'
    location = (0.6, -0.8, 0.6) # (UI: click mesh > Transform > Location)
    rotation = (90, 0, 120) # (UI: click mesh > Transform > Rotation)
    scale = (1.3,1.3,1.3) # (UI: click mesh > Transform > Scale)
    mesh = bt.readMesh(meshPath, location, rotation, scale)

    # # set material 
    #ptColor = bt.colorObj(bt.derekBlue, 0.5, 1.3, 1.0, 0.0, 0.0)

    RGBA =   (144.0/255, 210.0/255, 236.0/255, 1)

    #RGBA =  (210.0/255, 144.0/255, 236.0/255, 1)
    ptColor = bt.colorObj(RGBA, 0.5, 1.3, 1.0, 0.0, 0.0)
    
    ptSize = 0.01
    bt.setMat_pointCloud(mesh, ptColor, ptSize)

    ## set invisible plane (shadow catcher)
    #bt.invisibleGround(shadowBrightness=0.9)

    ## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
    camLocation = (3, 0, 2)
    lookAtLocation = (0,0,0.5)
    focalLength = 45 # (UI: click camera > Object Data > Focal Length)
    cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

    ## set light
    lightAngle = (-30, -30, 155) 
    strength = 2
    shadowSoftness = 0.3
    sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

    ## set ambient light
    bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

    ## set gray shadow to completely white with a threshold 
    bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

    ## save blender file so that you can adjust parameters in the UI
    bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')

    ## save rendering
    bt.renderImage(outputPath, cam)
