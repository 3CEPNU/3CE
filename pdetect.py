import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils import *

import cv2
from myUtils import *
from tracking.tracking import Tracking
from tracking.unit_object import UnitObject
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import plot_one_box
import os
from datetime import date
from datetime import datetime
import math
import threading
os.environ['KMP_DUPLICATE_LIB_OK']='True'

ccc = 0
ratio = 0
thumb = 0
sum1 = 0
cross_car = [0]
idc = []
count = 0

class information:
    def __init__(self,id1,t1=0.1,t2=0.2,c1_x=0,c1_y=0,c2_x=0,c2_y=0,speed=0,save_count=0):
        self.id1 = id1
        self.t1 = t1
        self.t2 = t2
        self.c1_x = c1_x
        self.c1_y = c1_y
        self.c2_x = c2_x
        self.c2_y = c2_y
        self.speed = speed
        self.save_count = save_count

    def cal(self):
      speed = ratio * math.sqrt(math.pow(self.c2_x - self.c1_x, 2) + math.pow(self.c2_y - self.c1_y, 2)) / ((self.t2-self.t1)/3600)
      print('time check : ' + str(self.t2-self.t1))
      return speed

    def save(self):
        self.save_count = self.save_count + 1

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    global ratio
    
    today = date.today()
    #cf = open('counttxt/' + today.isoformat()+'.txt', 'a')
    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    tracker = Tracking()

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        bboxes = []
        coordinates = []
        print("\n")

        #onemincheck()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                cv2.line(im0,(0,300),(1250,300),(0,0,255),2)
                cv2.line(im0,(0,350),(1250,350),(0,0,255),2)

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        print(label)
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                        bboxes.append([int(cls), conf, label, c1, c2])

                    if(names[int(cls)] == 'person'):
                        t3 = time.time()
                        if(t3 - t0 < 20.4 and t3 - t0 > 20):
                            first_check = (xyxy[1])
                        if(t3 - t0 < 40.4 and t3 - t0 > 40):
                            second_check = (xyxy[1])
                            ratio = 0.0003 / float((second_check - first_check))
                            tf = open('ratio.txt', 'w')
                            tf.write(str(ratio))
			    


            bboxes = delete_overlappings(bboxes, 0.8)
            for box in bboxes:
                #  coordinates.append(UnitObject( [box[3][0], box[3][1], box[4][0], box[4][1] ], box[0]))
                coordinates.append(UnitObject( [box[3][0], box[3][1], box[4][0], box[4][1] ], 1))
                #print(box)
            

            tracker.update(coordinates)
            for j in range(len(tracker.tracker_list)):
                #print("===============================\n")
                x = int(tracker.tracker_list[j].unit_object.box[0])
                y = int(tracker.tracker_list[j].unit_object.box[1])
                cv2.putText(im0,str(tracker.tracker_list[j].tracking_id), (x,y),0, 1, (0,0,255),2)
                #print("tracker(%d) %d %d  hits: %d" %(j,x,y, tracker.tracker_list[j].hits))
                ######################################################################################################################################################################
                c_x = int((int(tracker.tracker_list[j].unit_object.box[0]) + int(tracker.tracker_list[j].unit_object.box[2]))/2)
                c_y = int((int(tracker.tracker_list[j].unit_object.box[1]) + int(tracker.tracker_list[j].unit_object.box[3]))/2)
                y1 = int(tracker.tracker_list[j].unit_object.box[1])
                y2 = int(tracker.tracker_list[j].unit_object.box[3])
            
                
                global thumb
                global sum1
                global idc
                global cross_car
                global count
                global ccc

                diff = c_y - 350
                diff2 = c_y - 300
                
                if(c_y > 300):
                    idc.append(tracker.tracker_list[j].tracking_id)
            
                idc = list(set(idc))
        
                ####좌표
                if(diff2 > 0 and diff2 < 30): ###첫번째 라인 지날때 좌표와 시간
                    try:
                        if(globals()['ce' + str(tracker.tracker_list[j].tracking_id)].t1 > 0.1):
                            pass
                    except:
                        globals()['ce' + str(tracker.tracker_list[j].tracking_id)] = information(tracker.tracker_list[j].tracking_id)

                    if(globals()['ce' + str(tracker.tracker_list[j].tracking_id)].t1 == 0.1):
                        globals()['ce' + str(tracker.tracker_list[j].tracking_id)].c1_x = c_x
                        globals()['ce' + str(tracker.tracker_list[j].tracking_id)].c1_y = c_y
                        globals()['ce' + str(tracker.tracker_list[j].tracking_id)].t1 = datetime.now().timestamp()
                        print('hello : ' + str(globals()['ce' + str(tracker.tracker_list[j].tracking_id)].id1) + '\t' + str(globals()['ce' + str(tracker.tracker_list[j].tracking_id)].t1))
            
                if(diff > 0 and diff < 30): ###두번째 라인 지날때 좌표와 시간
                    try:
                        if (globals()['ce' + str(tracker.tracker_list[j].tracking_id)].t2 == 0.2):
                            globals()['ce' + str(tracker.tracker_list[j].tracking_id)].c2_x = c_x
                            globals()['ce' + str(tracker.tracker_list[j].tracking_id)].c2_y = c_y
                            globals()['ce' + str(tracker.tracker_list[j].tracking_id)].t2 = datetime.now().timestamp()

                    except KeyError:
                        print( 'not 1st line, but 2nd line' )
                #########################################################################################################################################################

            if view_img:
                cv2.waitKey(1)      

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow("img", im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    tf.close()
    cf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=240, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect()
                create_pretrained(opt.weights, opt.weights)
        else:
            detect()

