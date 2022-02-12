from atexit import register
from copy import Error
import cv2
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from numpy.ma.core import right_shift
from utils import MyZoom

class Mydetector:
    def __init__(self,setname = 'vegetables',config="configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml",
                    model="model/model_final.pth") -> None:
        self.metadata = self.register(setname)
        self.cfg = self.setup(config,model)
    def register(self,setname):
        self.valsetname = setname + '_val'

        register_coco_instances(self.valsetname, {}, 
                            '../dataset/'+setname+'/coco/val.json', 
                        '../dataset/'+setname+'/val')
        MetadataCatalog.get(self.valsetname).thing_classes=['cucumber', 'eggplant', 'pepper', 'tomato']
        MetadataCatalog.get(self.valsetname).thing_dataset_id_to_contiguous_id={0: 0, 1: 1, 2: 2, 3: 3}
        MetadataCatalog.get(self.valsetname).keypoint_names = ["handle", "top", "left","bottom","right","center"]
        MetadataCatalog.get(self.valsetname).keypoint_flip_map = [("left", "right")]
        MetadataCatalog.get(self.valsetname).keypoint_connection_rules = []

        coco_val_metadata = MetadataCatalog.get(self.valsetname)
        return coco_val_metadata

    def setup(self,config,model):
        cfg = get_cfg()
        cfg.merge_from_file(config)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # coco datasets
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 6
        cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        cfg.MODEL.WEIGHTS = model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
        return cfg

    def revert(self,points,mz: MyZoom):
        centerx = mz.oriwidth / 2
        centery = mz.oriheight / 2
        for i in range(len(points['instances'].pred_boxes)):
            newboxminx = centerx + (points['instances'].pred_boxes.tensor[i][0]-centerx) / mz.multiple
            points['instances'].pred_boxes.tensor[i][0] = newboxminx
            newboxminy = centery + (points['instances'].pred_boxes.tensor[i][1]-centery) / mz.multiple
            points['instances'].pred_boxes.tensor[i][1] = newboxminy
            newboxmaxx = centerx + (points['instances'].pred_boxes.tensor[i][2]-centerx) / mz.multiple
            points['instances'].pred_boxes.tensor[i][2] = newboxmaxx
            newboxmaxy = centery + (points['instances'].pred_boxes.tensor[i][3]-centery) / mz.multiple
            points['instances'].pred_boxes.tensor[i][3] = newboxmaxy
            for j in range(6):
                newpointx = centerx + (points['instances'].pred_keypoints[i][j][0]-centerx) / mz.multiple
                points['instances'].pred_keypoints[i][j][0] = newpointx
                newpointy = centery + (points['instances'].pred_keypoints[i][j][1]-centery) / mz.multiple
                points['instances'].pred_keypoints[i][j][1] = newpointy
        return points

    def predictor(self, mz: MyZoom):
        predictor = DefaultPredictor(self.cfg)
        outputs = predictor(mz.oriimg)
        reout = outputs
        maxscores = sum(outputs['instances'].scores)
        
        for i in range(10):
            outputs = predictor(mz.zoom_in(1.1))
            sumscores = sum(outputs['instances'].scores)
            if sumscores > maxscores:
                maxscores = sumscores
                reout = self.revert(outputs,mz)
        return reout

    def visual(self,img,outputs):
        v = Visualizer(img[:, :, ::-1],
                        metadata=self.metadata, 
                        scale=2.0, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return v.get_image()[:, :, ::-1]

    def evaluator(self):
        predict = DefaultPredictor(self.cfg)
        dataset_dicts = DatasetCatalog.get(self.valsetname)
        RightNum = [0,0,0,0]
        ErrorD = [0,0,0,0]
        ErrorL = [0,0,0,0]
        for d in dataset_dicts:
            img = cv2.imread(d['file_name'])
            outputs = predict(img)['instances'].to("cpu")
            preclass = int(outputs.get_fields()['pred_classes'][0])
            if preclass == d['annotations'][0]['category_id']:
                RightNum[preclass] += 1
            
                PredTop = outputs.get_fields()['pred_keypoints'][0][1].numpy()
                RealTop = d['annotations'][0]['keypoints'][3:6]
                PredLeft = outputs.get_fields()['pred_keypoints'][0][2].numpy()
                RealLeft = d['annotations'][0]['keypoints'][6:9]
                PredBottom = outputs.get_fields()['pred_keypoints'][0][3].numpy()
                RealBottom = d['annotations'][0]['keypoints'][9:12]
                PredRight = outputs.get_fields()['pred_keypoints'][0][4].numpy()
                RealRight = d['annotations'][0]['keypoints'][12:15]
                
                Predd = ((PredLeft[0]-PredRight[0])**2 + (PredLeft[1]-PredRight[1])**2)**0.5
                Reald = ((RealLeft[0]-RealRight[0])**2 + (RealLeft[1]-RealRight[1])**2)**0.5
                derr = ((Predd-Reald)/Reald)**2
                ErrorD[preclass] += derr

                Predl = ((PredTop[0]-PredBottom[0])**2 + (PredTop[1]-PredBottom[1])**2)**0.5
                Reall = ((RealTop[0]-RealBottom[0])**2 + (RealTop[1]-RealBottom[1])**2)**0.5
                lerr = ((Predl-Reall)/Reall)**2
                ErrorL[preclass] += lerr

        ed = [(a/b)**0.5 for a,b in zip(ErrorD,RightNum)]
        el = [(a/b)**0.5 for a,b in zip(ErrorL,RightNum)]  
        
        return RightNum, ed, el
