#License Terms

#Copyright (c) 2020-21, California Institute of Technology ("Caltech").  U.S. Government sponsorship acknowledged.

#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#* Redistributions must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#* Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



# Dependencies
import numpy as np
from libtiff import TIFF
import os
import sys
import time
import math
import tensorflow as tf
from tensorflow import keras
import multiprocessing
import random
from PIL import Image

def rescale_generic(img_input, scale):
    sh = list(img_input.shape)
    rsh = (int(np.ceil(sh[0] / scale)), int(np.ceil(sh[1] / scale)))
    if (rsh[0] * scale > sh[0] or rsh[1] * scale > sh[1]):
        img = np.zeros((rsh[0] * scale, rsh[1] * scale), img_input.dtype)
        img[:sh[0],:sh[1]] = img_input
        for i in range(0, (sh[0] // scale) + 1):
            img[i*scale:(i+1)*scale,sh[1]:rsh[1]*scale] = np.average(img_input[i*scale:(i+1)*scale,(rsh[1] - 1)*scale:sh[1]])
        for i in range(0, (sh[1] // scale) + 1):
            img[sh[0]:rsh[0]*scale,i*scale:(i+1)*scale,] = np.average(img_input[(rsh[0] - 1)*scale:sh[0],i*scale:(i+1)*scale])
        img[(rsh[0] - 1)*scale:rsh[0]*scale,(rsh[1] - 1)*scale:rsh[1]*scale] = np.average(img_input[(rsh[0] - 1)*scale:sh[0],(rsh[1] - 1)*scale:sh[1]])
        img[(rsh[0] - 1)*scale:sh[0],(rsh[1] - 1)*scale:sh[1]] = img_input[(rsh[0] - 1)*scale:sh[0],(rsh[1] - 1)*scale:sh[1]]
    else:
        img = img_input

    # https://scipython.com/blog/binning-a-2d-array-in-numpy/
    shape = (rsh[0], scale,
             rsh[1], scale)
    return img.reshape(shape).mean(-1).mean(1)

def rescaleDown(lims,data):
    return np.interp(data,lims,(-1,1))

def rescaleUp(lims,data):
    return np.interp(data,(-1,1),lims)

def rescaleDown_Enhance(lims,data):
    return np.interp(data,lims,(-1,1))

def rescaleUp_Enhance(lims,data):
    return data

def openDem(path):
    print("[ LOG ] Opening DEM... " + path)
    return TIFF.open(path).read_image()

def getMola(latu,latl,lonl,lonr,mola):
    m = mola[128*(90-latu)+0:128*(90-latl)+0,128*(180+lonl)+1:128*(180+lonr)+1]
    if m.shape[1] < 512:
        new = np.zeros((m.shape[0], 512))
        new[:,:m.shape[1]] = m[:,:]
        for x in range(m.shape[1], 512):
            new[:,x] = mola[128*(90-latu)+0:128*(90-latl)+0,x - m.shape[1]]
        m = new
    if m.shape[0] == 511:
        new = np.zeros((512, 512))
        new[:511,:] = m[:,:]
        new[511,:] = m[510,:]
        m = new
    return m.astype(np.int16)

def getMolaWithPadding(latu,latl,lonl,lonr,mola,padding):
    prepad = padding[0]
    postpad = padding[1]
    pad = prepad + postpad
    coords = (128*(90-latu)-prepad,128*(90-latl)+postpad,128*(180+lonl)+1-prepad,128*(180+lonr)+1+postpad)
    xsize = mola.shape[1]-1
    ysize = mola.shape[0]-1
    xtilesize = 511
    ytilesize = 511
    ccoords = (max(coords[0], 0), min(coords[1], ysize), max(coords[2], 0), min(coords[3], xsize))
    diff = (ccoords[0]-coords[0], coords[1]-ccoords[1], ccoords[2]-coords[2], coords[3]-ccoords[3])
    result = np.zeros((ytilesize+1+pad,xtilesize+1+pad))
    result[diff[0]:ytilesize+1+pad-diff[1],diff[2]:xtilesize+1+pad-diff[3]] = mola[ccoords[0]:ccoords[1],ccoords[2]:ccoords[3]]
    if diff[3] > 0:
        result[diff[0]:ytilesize+1+pad-diff[1],-diff[3]:] = mola[ccoords[0]:ccoords[1],0:diff[3]]
    return result

def getCTXFilename(dirs, lat, lon):
    for directory in dirs:
        for prefix in ["Murray-", "Murray"]:
            fileName = directory + prefix + "Lab_CTX-Mosaic_beta01_E"+str(lon).zfill(3+int(0.5-0.5*np.sign(lon)))+"_N"+str(lat).zfill(2+int(0.5-0.5*np.sign(lat)))+".tif"
            if os.path.isfile(fileName):
                return fileName

def getCTX(args):
    (lat, lon) = args
    # return a 2 degree square tile of CTX data referenced by lat and lon.
    file = getCTXFilename(["../data/zips/"], lat, lon)
    print('[ LOG ] Opening CTX file {}'.format(file))
    try:
        img = TIFF.open(file).read_image()
        if img[1,0] == 0 and img[2,0] == 0 and img[3,0] == 0 and img[1,1] > 0 and img[2,1] > 0 and img[3,1] > 0:
            x = img[:,1]
            img[:,0] = x
        if img[0,1] == 0 and img[0,2] == 0 and img[0,3] == 0 and img[1,1] > 0 and img[1,2] > 0 and img[1,3] > 0:
            x = img[1, :]
            img[0, :] = x
        return img.astype(np.uint8)
    except:
        print("ERROR: Could not find "+file+", inserted zeros.")
        return np.zeros((23710,23710))

def saveDem(path,dem,split=False):
    if not split:
        converted = dem.astype(np.float32)
        tif = TIFF.open(path+".tif", mode='w')
        tif.write_image(converted)
    else:
        sf = int(np.ceil(dem.shape[0]/8192))
        size = int(dem.shape[0]/sf)
        for i in range(sf):
            for j in range(sf):
                tif = TIFF.open(path+"_"+str(i)+"_"+str(j)+".tif", mode='w')
                tif.write_image(dem[i*size:(i+1)*size,j*size:(j+1)*size])

def ctxS(data):
    return rescaleDown_Enhance((0,255), data)

def interpPiece(model,im,rescale_lims,dim):
    # im has twice resolution of dem, matches model trained shape
    # Try with generators and multiprocessing in model.predict
    #   https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    inp = np.zeros((1,2*dim,2*dim,1))
    inp[0,:,:,0] = ctxS(im)
    return rescaleUp_Enhance(rescale_lims, model.predict(inp)[0,:,:,0])

def interpBigImg_opt(model,ctx,upscale_lims,padding,dim):
    sT = time.time()
    csh = list(ctx.shape)
    # Pad output by the useless margins.
    upad = padding[0]
    pad = padding[1]
    csh[0]+=pad+upad
    csh[1]+=pad+upad
    # Construct inputs
    ctxInput = np.zeros(csh)
    ctxInput[upad:-pad,upad:-pad]=ctx
    for i in range(pad):
        ctxInput[-pad+i] = ctxInput[-pad-1+i]
        ctxInput[:,-pad+i] = ctxInput[:,-pad-1+i]
    for i in range(upad):
        ctxInput[upad-i-1] = ctxInput[upad-i]
        ctxInput[:,upad-i-1] = ctxInput[:,upad-i]
    # Specify output container
    output = np.zeros(csh)
    # fill in, then do right and bottom edges.
    # dim is the size of the model mola input square, ie half of the ctx input.
    stride = 2*dim-pad-upad

    num_samples = (csh[1] - 2 * dim) // stride + 1

    for i in range(0,csh[0]-2*dim,stride):    
        batch = np.zeros((num_samples, 2 * dim, 2 * dim, 1))
        for j in range(0,csh[1]-2*dim,stride):
            batch[j//stride,:,:,0] = ctxS(ctxInput[i:i+2*dim,j:j+2*dim])
        res = model.predict_on_batch(batch)
        for j in range(0,csh[1]-2*dim,stride):
            output[i:i+2*dim,j:j+2*dim][upad-1:-pad+1,upad-1:-pad+1] = rescaleUp_Enhance(upscale_lims, res[j//stride,:,:,0])[upad-1:-pad+1,upad-1:-pad+1]
    
    batch = np.zeros((2, num_samples, 2 * dim, 2 * dim, 1))
    for i in range(0,csh[0]-2*dim,stride):
        batch[0,i//stride,:,:,0] = ctxS(ctxInput[i:i+2*dim,-2*dim:])
        batch[1,i//stride,:,:,0] = ctxS(ctxInput[-2*dim:,i:i+2*dim])
    res0 = model.predict_on_batch(batch[0,:,:,:,])
    res1 = model.predict_on_batch(batch[1,:,:,:,])
    for i in range(0,csh[0]-2*dim,stride):
        output[i:i+2*dim,-2*dim:][upad:-pad,upad:-pad] = rescaleUp_Enhance(upscale_lims, res0[i//stride,:,:,0])[upad:-pad,upad:-pad]
        output[-2*dim:,i:i+2*dim][upad:-pad,upad:-pad] = rescaleUp_Enhance(upscale_lims, res1[i//stride,:,:,0])[upad:-pad,upad:-pad]

    output[-2*dim:,-2*dim:][upad:-pad,upad:-pad] = interpPiece(model, ctxInput[-2*dim:,-2*dim:], upscale_lims, dim)[upad:-pad,upad:-pad]
    
    print("[ LOG ] interpBigImg " + str(int(time.time() - sT)) + " sec")
    return output[upad:-pad,upad:-pad]

def enhanceMain(pool, mola, lat, lon, res, model_enhance = None):
    # defines 4 square degree graticule of given planet defined by lat, lon. Top left corner. 
    # Enhances until resolution better than res. User expected to know limits of imagery, around 5 m for CTX. 
    tileSize = 4 # 4 degrees x 4 degree tiles, corresponding to Murray lab CTX data
    dim = 32 # dimension of origin DEM tile, image and output are 2*dim.
    planetRadius = 3389500 #m
    
    print("[ LOG ] Planetary DEM enhancement tool operating on " + str(tileSize) + " degree tile from")
    print("[ LOG ] latitude: [" + str(lat) + "," + str(lat-tileSize) + "], longitude: [" + str(lon) + "," + str(lon+tileSize)+"].")
    
    initialRes = 2 * np.pi * 3389500 / mola.shape[1]
    print("[ LOG ] Initial DEM resolution "+str(initialRes)+" m.")
    numEnhance = int(np.ceil(np.log2(462 / res)))
    print("[ LOG ] Will enhance " + str(numEnhance)+" time(s), a total resolution increase of "+str(2**numEnhance)+" to " +
          str(initialRes/2**numEnhance) + " m.")

    demRes = getMola(lat,lat-tileSize,lon,lon+tileSize,mola).shape[0]

    molaTrain = getMola(lat,lat-tileSize,lon,lon+tileSize,mola)

    # Debug images
    #tif = TIFF.open("interp_mola.tif", mode='w')
    #tif.write_image((molaTrain + 9000).astype(np.int16))
    #tif.close()

    print("[ LOG ] Initializing scaling helper functions.")
    # Rescale data to within 0,1 for machine learning stuff. Leave some room for peaks.
    molaMin = np.min(molaTrain)
    molaMax = np.max(molaTrain)
    molaLims = (1.02 * molaMin - 0.02 * molaMax, 1.02 * molaMax - 0.02 * molaMin)

    print("[ LOG ] Generating interp error tile...")
    interpPadding=[4, 4]
    molaTrainSmall = rescale_generic(molaTrain.astype(np.float32), 2)
    molaTrainInterp = np.array(Image.fromarray(molaTrainSmall.astype(np.float32)).resize(size=molaTrain.shape, resample=Image.Resampling.BICUBIC))
    molaTrainDiff = (molaTrain - molaTrainInterp)
    
    # Add to scaling functions
    mola2absMax = np.max([np.abs(np.min(molaTrainDiff[3:-3,3:-3])), np.abs(np.max(molaTrainDiff[3:-3,3:-3]))])
    mola2Min = -mola2absMax
    mola2Max = mola2absMax
    mola2lims = np.array([mola2Min,mola2Max])
    
    data = -np.sort(-(molaTrainDiff[3:-3,3:-3].flatten()))
    absMax = np.max([np.abs(data[56]), np.abs(data[-57])])
    molaTrainDiff[molaTrainDiff > absMax] = absMax
    molaTrainDiff[molaTrainDiff < -absMax] = -absMax
    print('[ LOG ] scaling before adjustment {}'.format(absMax))
    absMax *= min(3.0, 1.0 / math.cos(math.radians(lat - 2.0)))
    if absMax < 48.0:
        absMax = 48.0
    elif absMax > 104.0:
        absMax = 104.0
    print('[ LOG ] scaling after adjustment  {}'.format(absMax))
    trainLims=np.array([-absMax,absMax])
    mola2lims=trainLims

    print("[ LOG ] Enchance training tile bounds " + str(mola2lims))

    # Get ctx
    print("[ LOG ] Opening CTX...")
    sT = time.time()
    args = []
    for latitude in range(lat-2, lat-2-tileSize, -2):
        for longitude in range(lon, lon+tileSize, 2):
            args.append([latitude, longitude])
    data = pool.map(getCTX, args)
    ctx = np.concatenate([
            np.concatenate([data[y*2+x]
                for y in (0, 1)],axis=0)
                for x in (0, 1)],axis=1)
    print("[ LOG ] Loaded CTX in " + str(int(time.time() - sT)) + " sec")
    
    # Avoid creating sharp edges around areas with missing data
    ctx[ctx == 0] = 127

    # Scale the ctx values to have more details in the enhanced image
    ctxSmall = np.array(Image.fromarray(ctx).resize(size=(8192, 8192), resample=Image.Resampling.NEAREST))
    h, edges = np.histogram(ctxSmall, range(257))
    x = np.cumsum(-np.sort(-h))
    pixels = ctxSmall.shape[0] * ctxSmall.shape[1]
    idx = np.nonzero(x > (pixels * 0.9))[0][0]
    threshold = 0.9998
    print('[ LOG ] 90%: {}   95%: {}   99.97%: {}'.format(np.nonzero(x > (pixels * 0.9))[0][0], np.nonzero(x > (pixels * 0.95))[0][0], np.nonzero(x > (pixels * threshold))[0][0]))
    ctxScaling = 220.0 / min(220.0, np.nonzero(x > (pixels * threshold))[0][0])
    print('[ LOG ] CTX scaling: {}'.format(ctxScaling))
    
    ctx = ctx.astype(np.float32)
    ctx[:] -= 127.5
    ctx[:] *= np.max([1.0, ctxScaling])
    ctx[:] += 127.5
    ctx[ctx < 0.0] = 0.0
    ctx[ctx > 255.0] = 255.0
    ctx = ctx.astype(np.uint8)

    def resizeCTX(factor, resampling_algo):
        img = Image.fromarray(ctx)
        return np.array(img.resize(size=(factor*demRes, factor*demRes), resample=resampling_algo))

    # Enhance until resolution condition met
    sT = time.time()
    details = np.zeros(molaTrain.shape).astype(np.float32)
    # Taking an extra margin around the MOLA tile to avoid incorrect gradient around the edges after enlargement
    demBase = getMolaWithPadding(lat,lat-tileSize,lon,lon+tileSize,mola, (1, 1))
    noise_amplitude = 4.8
    padding = [13, 11, 9, 7, 4, 4, 4]
    for i in range(numEnhance):
        currentResolution = initialRes / 2**(i+1)
        print('[ LOG ] Enhancement {} of {}.'.format(i+1, numEnhance))
        print('[ LOG ] Current resolution {} m.'.format(currentResolution))
        if i < numEnhance - 2:
            resample_algo = Image.Resampling.BILINEAR
        else:
            resample_algo = Image.Resampling.LANCZOS
        enhancedCTX = resizeCTX(2**(i+1), resample_algo)
        currentDetails = interpBigImg_opt(model_enhance, enhancedCTX, [-1.0, 1.0], [padding[i], padding[i]], dim)
        # Normalize to the full range
        adjustment = mola2lims[1]
        adjustment /= np.max([abs(np.min(currentDetails)), abs(np.max(currentDetails))])
        print('[ LOG ] Scaling factor {}   Details max {}'.format(int(mola2lims[1]), np.max([abs(np.min(currentDetails)), abs(np.max(currentDetails))])))
        details = np.interp(currentDetails, (-1,1), (-adjustment, adjustment)) + np.array(Image.fromarray(details).resize(size=currentDetails.shape, resample=Image.Resampling.BILINEAR))

        # Add some noise to the base DEM to mask weird artefacts
        if i < 4:
            demBase = demBase + np.random.random(demBase.shape) * noise_amplitude - noise_amplitude / 2
        demBase = np.array(Image.fromarray(demBase).resize(size=(demBase.shape[0]*2, demBase.shape[1]*2), resample=Image.Resampling.BILINEAR))

        noise_amplitude *= 0.5
        mola2lims *= 0.5

    x = (demBase.shape[0] - details.shape[0]) // 2
    demBase = demBase[x:-x,x:-x]
    
    # Save output as .tiff
    maxRes = 2**(8+numEnhance)
    if demBase.shape[0] > maxRes:
        details = np.array(Image.fromarray(details).resize(size=(maxRes, maxRes), resample=Image.Resampling.LANCZOS))
        demBase = np.array(Image.fromarray(demBase).resize(size=(maxRes, maxRes), resample=Image.Resampling.BILINEAR))

    enhancedDem = demBase + details

    print("[ LOG ] Saving enhanced DEM")
    currentResolution = initialRes / 2**(numEnhance-1)
    saveDem("../output/enhanced_dem_"+str(lat)+"_"+str(lon)+"_"+str(int(currentResolution))+"_m",enhancedDem,False)

    print("[ LOG ] Enhance took " + str(int(time.time() - sT)) + " sec\n")


# Set latitude band, lat at top of band
# Gale (-4, 136)
# Jezero (20, 76)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Get lat from arg.
    lat = int(sys.argv[1])
    print("[ LOG ] Processing lat: " + str(lat))
    
    pool = multiprocessing.Pool(2)
    lonStart = int(sys.argv[2])
    if len(sys.argv) > 3:
        lonEnd = int(sys.argv[3])
    else:
        lonEnd = lonStart + 1

    try:
        mola = openDem("f:/programming/resources/mars/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif")
        model_dir = "f:/programming/resources/mars/DEM Enhancement/models/model_enhance_-4"
        if os.path.isdir(model_dir):
            model_enhance = keras.models.load_model(model_dir)
            print(model_enhance.summary())
            for lon in range(lonStart, lonEnd, 4):
                enhanceMain(pool, mola, lat, lon, 24, model_enhance)
        else:
            print('[ LOG ] Enahance model cannot be read from {}'.format(model_dir))
        
    except RuntimeError as e:
        print("FAILCODE: "+str(lat) + " " + str(lon))
        print(e)
    pool.close()
    pool.terminate()
