import matplotlib.pyplot as plt
import cv2

def cv2_imshow(img):
    implt = img[:,:,::-1]
    plt.figure(figsize=(15, 15))
    plt.imshow(implt)
    plt.show()

def depshow(depth,path = None):
    #deepmap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.5), cv2.COLORMAP_JET)
    plt.figure(figsize=(15, 15))
    plt.imshow(depth/100)
    plt.colorbar(fraction = 0.03)
    if path:
            plt.savefig(path)

    plt.show()

class MyZoom:
    def __init__(self,img) -> None:
        self.oriimg = img
        self.oriheight = img.shape[0]
        self.oriwidth = img.shape[1]
        self.multiple = 1
    
    def zoom_in(self,mul):
        img = self.oriimg
        self.multiple = self.multiple*mul
        if self.multiple < 1:
            self.multiple = 1
        height = (int)(self.oriheight*self.multiple)
        width = (int)(self.oriwidth*self.multiple)
        img = cv2.resize(img, (width,height), interpolation=cv2.INTER_LINEAR)
        miny = (int)((height-self.oriheight)/2)
        maxy = miny+self.oriheight
        minx = (int)((width-self.oriwidth)/2)
        maxx = minx+self.oriwidth
        crop_img = img[miny:maxy,minx:maxx]
        return crop_img

def measure(outputs, depth):
    PredTop = outputs.get_fields()['pred_keypoints'][0][1].numpy()
    PredLeft = outputs.get_fields()['pred_keypoints'][0][2].numpy()
    PredBottom = outputs.get_fields()['pred_keypoints'][0][3].numpy()
    PredRight = outputs.get_fields()['pred_keypoints'][0][4].numpy()
                
    Preddp = ((PredLeft[0]-PredRight[0])**2 + (PredLeft[1]-PredRight[1])**2)**0.5
    Predlp = ((PredTop[0]-PredBottom[0])**2 + (PredTop[1]-PredBottom[1])**2)**0.5

    Di = 1.05*Preddp*1.55/480
    Li = 1.09*Predlp*1.55/480

    Rd = depth*Di/(1.88-Di/2)
    Rl = depth*Li/(1.88-Di/2)

    return Rd,Rl

