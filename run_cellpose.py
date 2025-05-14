from glob import glob
import tifffile as tiff
from cellpose import models, io, transforms
from datetime import datetime as dt
import numpy as np
import sys
import os

def run_cellpose_model(path_model, dir_in, dir_out):
    '''run a cellpose model on images in dir_in folder and save the outputs in dir_out'''
    model = models.CellposeModel(gpu=True, pretrained_model=path_model)
    flist = glob(dir_in + '*.tif')
    print(flist)
    for f in flist:
        img = io.imread(f)
        returned_ = model.eval(img, diameter=None, flow_threshold=1) #, normalize={"tile_norm_blocksize":256,"percentile":[1,99]}
        output = np.zeros(returned_[0].shape)
        output[returned_[0]>0] = 3
        fname = f.split('\\')[-1]
        tiff.imwrite(dir_out + fname, output.astype(np.uint8))



neun_model_path = r"Z:\Labmembers\Ingvild\RM1\protocol_testing\202503_LargeBatch_AgingCCFBrains\pilot_analysis\561Neun\training_with_MIPs\training_chunks\models\05072025_NeuN-model"
input_sections = r"Z:\Labmembers\Ingvild\RM1\protocol_testing\202503_LargeBatch_AgingCCFBrains\pilot_analysis\561Neun\training_with_MIPs\images_for_validation\\"
out_path = r"Z:\Labmembers\Ingvild\RM1\protocol_testing\202503_LargeBatch_AgingCCFBrains\pilot_analysis\561Neun\training_with_MIPs\images_for_validation\model_predictions\\"
os.mkdir(out_path)

run_cellpose_model(neun_model_path, input_sections, out_path)