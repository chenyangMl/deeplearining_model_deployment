# -*- encoding: utf-8 -*-
import time
import torch
import numpy as np
import cv2
import os,sys
import argparse
import io
import requests
from pathlib import Path
from collections import OrderedDict, namedtuple
from easydict import EasyDict as edict
sys.path.insert(0, os.getcwd())


from mainareaDet.utils.general import polygon_non_max_suppression, polygon_scale_coords
from mainareaDet.util import sort_box
from mainareaDet.serverProcess import pre_server, post_server


class MainareaDet():
    """
    封装yolo的本地推理模块
    """
    def __init__(self, modelPath, netCfg, nc=2, **kargs) -> None:
        
        assert os.path.exists(modelPath), "Not found %s!!!"%modelPath
        assert os.path.exists(netCfg), "Not found %s!!!"%netCfg
        gpuId = kargs.get("gpuId", 0)
        fp16 = kargs.get("fp16", 0)
        device = torch.device("cuda:%s"%gpuId)
        modelType = modelPath.suffix[1:]
        self.modelType = modelType

        if modelType.endswith("pth"):
            self.model = self.load_model(modelPath, netCfg, nc, fp16)
            self.model.to(device).eval()
        elif modelType.endswith("engine"):
            self.loadengineModel(modelPath, device)
        
        self.fp16 = fp16
        self.pre_client =  self.pre_client
        self.post_client = self.post_client
        self.pre_server =  pre_server
        self.post_server = post_server



    def __call__(self, imgList, debug=0, dewarp=1):
        warpImgs, tMList = [None]*2

        
        imgList1, metaList = self.pre_client(imgList,padding=True) if self.modelType.endswith("engine") else self.pre_client(imgList)
        imgT = self.pre_server(imgList1, fp16=self.fp16)
        # t0 = time.time()
        preds1 = self.inference(imgT)
        # print("inferenceT=%.dms\n"%(time.time()-t0)*1000)
        preds1 = self.post_server(preds1)
        preds, debugImgs = self.post_client(preds1, imgList, imgList1, metaList, debug=debug)
        if dewarp:
            warpImgs, tMList = self.mainareaDewarp(preds, imgList)
        
        return edict({
            "preds"      : preds,
            "debugImgs"  : debugImgs,
            "warpImgs"   : warpImgs,
            "tMList"     : tMList
        })


        
    def inference(self, imgT, **kargs):
        if self.modelType.endswith("pth"): # pytorch
            print("[Pytorch]: inference using pytorch.")
            with torch.no_grad():
                preds = self.model(imgT)
            return preds
        elif self.modelType.endswith("engine"): # tensorrt
            preds = []
            if imgT.shape != self.bindings["images"].shape:   # 动态batch推理，需要模型转换支持动态batch才可以
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, imgT.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=imgT.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert imgT.shape == s, f"input size {imgT.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(imgT.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            preds = [self.bindings[x].data for x in sorted(self.output_names)]
            return preds

    def pre_client(self, imgList, inSize=224, auto=True, scaleFill=False, scaleup=False, stride=32, padding=False, debug=0):
        new_shape = (inSize, inSize)
        imgList1, metaList = [], []
        for img0 in imgList:
            shape0 = img0.shape[:2]
            # Scale ratio (new / old)
            r = min(new_shape[0] / shape0[0], new_shape[1] / shape0[1])
            if not scaleup:  # only scale down, do not scale up (for better val mAP)
                r = min(r, 1.0)
            # Compute padding
            ratio = r, r  # width, height ratios
            new_unpad = int(round(shape0[1] * r)), int(round(shape0[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
            if auto:  # minimum rectangle
                dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
            elif scaleFill:  # stretch
                dw, dh = 0.0, 0.0
                new_unpad = (new_shape[1], new_shape[0])
                ratio = new_shape[1] / shape0[1], new_shape[0] / shape0[0]  # width, height ratios

            dw /= 2  # divide padding into 2 sides
            dh /= 2

            if shape0[::-1] != new_unpad:  # resize
                img0 = cv2.resize(img0, new_unpad, interpolation=cv2.INTER_LINEAR)
            if padding:
                target = new_shape[0]
                h, w = img0.shape[:2]
                if h < target:
                    dh = (target - h) /2
                elif w < target:
                    dw = (target - w) /2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img1 = cv2.copyMakeBorder(img0, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))  # add border
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            # print("img0.shape=%s, img1.shape=%s"%(img0.shape, img1.shape))
            imgList1.append(img1); metaList.append((ratio, (dw, dh)))

        return imgList1, metaList

    def post_client(self, preds, imgList, imgList1, metaList, debug=1, conf_thres=0.6, iou_thres=0.5, **kargs):
        """对模型的预测结果进行后处理"""

        preds0, debugImgs = [], [] #img1坐标转换到原图img0坐标
        preds = polygon_non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres, 
                classes=None, agnostic=False, multi_label=True, max_det=100)
        for det, img0, img1, meta in zip(preds, imgList, imgList1, metaList):
            # NMS
            
            debugImg = img0.copy()
            # vaildMask = det[:,-1] == 1  #仅仅保留classid=1的bbox
            # det = det[vaildMask]
            if len(det):
                # Rescale boxes from img_size to im0 size
                pred_poly = polygon_scale_coords(img1.shape[:2], det[:, :8], img0.shape[:2], ratio_pad=meta).round()
                det = torch.cat((pred_poly, det[:, 8:]), dim=1) # (n, [poly conf cls])
                det = det.cpu().numpy()
                preds0.append(det)
                if debug:
                    for bboxPred in reversed(det):
                        polygon, conf, cls  = bboxPred[:8], bboxPred[8], bboxPred[9]
                        polygon = polygon.reshape(4,2).astype(np.int32)
                        cv2.polylines(debugImg, [polygon], True, (0,0,255), 2)
                        cv2.putText(debugImg, "%d, %.2f"%(cls, conf), tuple(polygon[0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2)
                    debugImgs.append(debugImg)
        return preds0, debugImgs

    def load_model(self, modelPath, netCfg, nc=2, fp16=0):
        from mainareaDet.models.yolo import Polygon_Model
        model = Polygon_Model(cfg=netCfg, ch=3, nc=nc)
        statedict0 = torch.load(modelPath)
        model.load_state_dict(statedict0)
        model.eval()
        if fp16:
            model.half()
        return model

    def loadengineModel(self, modelPath, device=torch.device("cuda:0")):
        print(f'Loading {modelPath} for TensorRT inference...')
        import tensorrt as trt
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(modelPath, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr())) #可以理解成推理前先进行了数据占位。
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        self.__dict__.update(locals())  # assign all variables to self

    @staticmethod
    def mainareaDewarp(preds, imgList):
        """warp Crop from img
        """
        tMatrix = np.array([[1,0,0], [0,1,0], [0,0,1]])
        if len(preds) < 1 or len(preds[0]) < 1:
            return imgList, [tMatrix]
        warpImgs, tMList = [],[]
        for img0, pred in zip(imgList, preds):
            index = np.argmax(pred[:,8])        #这里取最大置信度的结果
            points0 = pred[index][:8].reshape(4,2)
            points0 = sort_box(points0)
            w = (np.linalg.norm(points0[1]-points0[0], 2) + np.linalg.norm(points0[3]-points0[2], 2)) // 2
            h = (np.linalg.norm(points0[2]-points0[1], 2) + np.linalg.norm(points0[3]-points0[0], 2)) // 2
            points1 = np.array([[0,0],[w,0], [w,h],[0,h]], np.float32)
            tMatrix = cv2.getPerspectiveTransform(points0, points1)
            warpImg = cv2.warpPerspective(img0, tMatrix, (int(w),int(h)))
            warpImgs.append(warpImg); tMList.append(tMatrix)
        return warpImgs, tMList

def loadImgs(imgdir):
    imgPaths   = glob(os.path.join(imgdir, "*.jpg"))
    imgs  = [cv2.imread(ipath) for ipath in imgPaths]
    print("Load %s images from path = %s"%(len(imgs), imgdir))
    return imgs, imgPaths



def main(args):
    args.modelPath = Path(args.modelPath).with_suffix(".%s"%args.modelType)
    assert args.modelPath.exists(), "Not found model path = %s"%args.modelPath

    mdObject  = MainareaDet(args.modelPath, args.modelCfg)
    imgs, imgPaths = loadImgs(args.imgdir)

    if args.batchInfer:
        outdirbatch = os.path.join(args.outdir, "batch_infer", args.modelType)
        if not os.path.exists(outdirbatch):
            os.makedirs(outdirbatch)
        predDict = mdObject(imgs, debug=1, dewarp=1)           #batch inference

        # --------------------------debug
        debugImg = np.hstack(predDict.debugImgs)           #输入是同样尺寸图片，不是的处理下就行
        warpImgs = predDict.warpImgs
        if warpImgs is not None:
            whs = np.array([img.shape[:2] for img in warpImgs])
            maxWH = np.max(whs, 0)
            dewarImgs = []
            for warpImg in warpImgs:
                warpImg_ = np.zeros((*maxWH,3), np.uint8)
                h1,w1 = warpImg.shape[:2]
                warpImg_[:h1,:w1] = warpImg
                dewarImgs.append(warpImg_)
            dewarpImg = np.hstack(dewarImgs)
            cv2.imwrite(os.path.join(outdirbatch, "dewarpImgs.jpg"), dewarpImg)
        cv2.imwrite(os.path.join(outdirbatch, "debugImgs.jpg"), debugImg)
        print("input >> %s  | output >> %s"%(args.imgdir, outdirbatch))

    else:
        outdirsingle = os.path.join(args.outdir, "single_infer", args.modelType)
        if not os.path.exists(outdirsingle):
            os.makedirs(outdirsingle)
        for img, ipath in zip(imgs, imgPaths):
            predDict = mdObject([img], debug=1, dewarp=1)    #single inference

            iname = ipath.split("/")[-1][:-4]
            warpPath = os.path.join(outdirsingle, "%s_1.jpg"%(iname))
            debugPath = os.path.join(outdirsingle, "%s_0.jpg"%(iname))
            cv2.imwrite(warpPath, predDict.warpImgs[0])
            cv2.imwrite(debugPath, predDict.debugImgs[0])
            print("input >> %s  | output >> %s"%(ipath, outdirsingle))

def argConfig():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--modelPath', type=str, default="weights/best.engine", help="加载的模型 (pth, onnx, engine)")
    parser.add_argument('--modelType',"--mt",  type=str, default="pth",choices=["engine", "pth"], help="使用什么网络结构进行推理")
    parser.add_argument('--modelCfg', type=str, default="mainareaDet/models/polygon_yolov5n.yaml", help="网络结构")
    parser.add_argument('--fp16', type=int, default=0, help="是否使用半精度")
    parser.add_argument('--imgdir', type=str, default="images", help="输入测试图目录")
    parser.add_argument('--outdir', type=str, default="runs", help="输出测试结果的目录")
    parser.add_argument('--batchInfer','--bi', type=int, default=0, help="batch推理测试")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """
        Usage:
        单图推理(默认pth模型):  python mainareaDet/main.py
        批量推理(默认pth模型):  python mainareaDet/main.py --bi 1
        批量推理(使用engine模型):  python mainareaDet/main.py --bi 1 --mt engine

    """

    from glob import glob
    args = argConfig()
    main(args)