import numpy as np
from libtiff import TIFF
import os
import sys
import time
import random
from scipy import ndimage
from PIL import Image, ImageFilter

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

def getCTXFilename(prefix, lat, lon):
    return prefix + "Lab_CTX-Mosaic_beta01_E"+str(lon).zfill(3+int(0.5-0.5*np.sign(lon)))+ \
           "_N"+str(lat).zfill(2+int(0.5-0.5*np.sign(lat)))+".tif"

def loadCTX(lat, lon):
    # return a 2 degree square tile of CTX data referenced by lat and lon.
    dire = "../data/zips/"
    file = getCTXFilename("Murray-", lat, lon)
    if not os.path.isfile(dire+file):
        file = getCTXFilename("Murray", lat, lon)
    try:
        img = TIFF.open(dire+file).read_image()
        if img[1,0] == 0 and img[2,0] == 0 and img[3,0] == 0 and img[1,1] > 0 and img[2,1] > 0 and img[3,1] > 0:
            x = img[:,1]
            img[:,0] = x
        if img[0,1] == 0 and img[0,2] == 0 and img[0,3] == 0 and img[1,1] > 0 and img[1,2] > 0 and img[1,3] > 0:
            x = img[1, :]
            img[0, :] = x
        return img.astype(np.uint8)
    except:
        print("ERROR: Could not find "+dire+file+", inserted zeros.")
        return np.zeros((23710,23710))

def loadCTXforTile(lat, lon, tileSizeDeg):
	print('[ LOG ] Opening CTX files for {} {}'.format(lat, lon))
	return np.concatenate([
            np.concatenate([
                loadCTX(lat,lon)
            for lat in range(lat-2,lat-2-tileSizeDeg,-2)],axis=0)
          for lon in range(lon,lon+tileSizeDeg,2)],axis=1)

def openDEM(path):
    print("[ LOG ] Opening DEM... " + path)
    return TIFF.open(path).read_image()

def getMolaTile(latu, latl, lonl, lonr, mola):
    m = mola[128*(90-latu)+0:128*(90-latl)+0,128*(180+lonl)+1:128*(180+lonr)+1]
    if m.shape[1] < 512:
        new = np.zeros((m.shape[0], 512))
        new[:,:m.shape[1]] = m[:,:]
        for i in range(0, 512 - m.shape[1]):
            new[:,512 - m.shape[1] + i] = mola[128*(90-latu)+1:128*(90-latl)+1,i]
        m = new
    if m.shape[0] == 511:
        new = np.zeros((512, 512))
        new[:511,:] = m[:,:]
        new[511,:] = m[510,:]
        m = new
    return m.astype(np.int16)

def getTrainingData(index, size, ctxTrain, molaTrainDiff):
    ctx = ctxTrain[index[0]:index[0]+size,index[1]:index[1]+size]
    if len(ctx[ctx==0]) > 0:
        return None
    diff = molaTrainDiff[index[0]:index[0]+size,index[1]:index[1]+size]
    return {
                'ctx':   ctx,
                'diff':  diff,
                'score': np.sum(np.abs(diff)),
                'x': index[1],
                'y': index[0]
            }

def saveTrainingItem(data, directory, lat, lon, x, y, suffix):
    converted = data.astype(np.float32)
    fname = 'train_{}_{}_{}_{}_{}.tif'.format(int(lat), int(lon), int(x), int(y), suffix)
    tif = TIFF.open(directory + fname, mode='w')
    tif.write_image(converted)

def trainingSetSort(x):
    return x['score']

def createTrainingSetForTile(mola, lat, lon, maxTrainingItems, outputDirectory):
    tileSizeDeg = 4
    dim = 32

    molaTrain = getMolaTile(lat, lat-tileSizeDeg, lon, lon+tileSizeDeg, mola)

    ctx = loadCTXforTile(lat, lon, tileSizeDeg)
    ctxTrain = np.array(Image.fromarray(ctx).resize(size=molaTrain.shape, resample=Image.BICUBIC))
    ctxMin = np.min(ctx)
    ctxTrain[ctxTrain < ctxMin] = 0

    molaTrainSmall = rescale_generic(molaTrain.astype(np.float32), 2)
    molaTrainInterp = np.array(Image.fromarray(molaTrainSmall.astype(np.float32)).resize(size=molaTrain.shape, resample=Image.BILINEAR))
    molaTrainDiff = molaTrain - molaTrainInterp
    data = np.sort((molaTrainDiff[3:-3,3:-3].flatten()))
    absMax = np.max([np.abs(data[15]), np.abs(data[-16])])
    molaTrainDiff[molaTrainDiff > absMax] = absMax
    molaTrainDiff[molaTrainDiff < -absMax] = -absMax
    absMax *= 1.03
    mola2lims = np.array([-absMax,absMax])
    print("[ LOG ] Enchance training tile bounds " + str(mola2lims))

    random_pairs = dict()
    for x in range(maxTrainingItems * 6):
        rp = np.random.randint(4, high = molaTrain.shape[0] - 2 * dim - 4, size = 2)
        random_pairs[molaTrain.shape[0] * (rp[0] // 1) + rp[1] // 1] = rp
    rl = [*random_pairs.values()]
    trainingSetRaw = [getTrainingData(rl[x], 2 * dim, ctxTrain, molaTrainDiff) for x in range(len(rl))]
    trainingSet = list(filter(None, trainingSetRaw))
    trainingSet.sort(reverse=False, key=trainingSetSort)
    trainingSet = trainingSet[maxTrainingItems * 5:]
    for i in range(len(trainingSet)):
        xData = np.interp(trainingSet[i]['ctx'], (0, 255), (-1, 1))
        saveTrainingItem(xData, outputDirectory, lat, lon, trainingSet[i]['x'], trainingSet[i]['y'], 'x')
        yData = np.interp(trainingSet[i]['diff'], mola2lims, (-1, 1))
        saveTrainingItem(yData, outputDirectory, lat, lon, trainingSet[i]['x'], trainingSet[i]['y'], 'y')

if __name__ == "__main__":
    latitude = -4
    directory = 'f:/programming/resources/mars/DEM Enhancement/trainingdata_{}/'.format(latitude)
    mola = openDEM("f:/programming/resources/mars/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif")
    for lon in range(-120, -40, 4):
        print('[ LOG ] Generating training set for tile latitude {} and longitude {}'.format(latitude, lon))
        createTrainingSetForTile(mola, latitude, lon, 2000, directory)
    print("[ LOG ] Finished generating training set for latitude" + str(latitude))
