import json, sys, cv2, numpy as np, onnxruntime as ort
from pathlib import Path
ROOT=Path("/Users/furb-x/Desktop/My Projects/BilimAI")
sess=ort.InferenceSession("/tmp/rp_weights/segm/segm_model.onnx", providers=["CPUExecutionProvider"]); name=sess.get_inputs()[0].name
gt=json.load(open(ROOT/"eval/testset_v1/ru_pages/ground_truth.json")); out={}
for fn in gt:
    img=cv2.imread(str(ROOT/"eval/testset_v1/ru_pages/images"/fn)); H,W=img.shape[:2]
    x=np.transpose(cv2.resize(img,(896,896)).astype(np.float32)/255,(2,0,1))[None]
    pred=sess.run(None,{name:x})[0][0]
    sx,sy=W/896,H/896
    # word boxes from shrinked_text
    m=(pred[0]>0.8).astype(np.uint8); cs,_=cv2.findContours(m,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    words=[]
    for c in cs:
        if cv2.contourArea(c)<10: continue
        x0,y0,w,h=cv2.boundingRect(c); words.append([x0*sx,y0*sy,(x0+w)*sx,(y0+h)*sy])
    # line polylines from text_line channel; assign each word to nearest line by vertical distance at the word's x
    ml=(pred[2]>0.5).astype(np.uint8); ml=cv2.dilate(ml,np.ones((3,3),np.uint8)); cl,_=cv2.findContours(ml,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    lines=[]
    for c in cl:
        if len(c)<5: continue
        pts=c[:,0,:].astype(np.float32); pts[:,0]*=sx; pts[:,1]*=sy; lines.append(pts)
    groups=[[] for _ in lines]
    for wb in words:
        cx,cy=(wb[0]+wb[2])/2,(wb[1]+wb[3])/2; best=None; bd=1e9
        for li,pts in enumerate(lines):
            near=pts[np.abs(pts[:,0]-cx)<max(30,(wb[2]-wb[0]))]
            if len(near)==0: continue
            d=np.min(np.abs(near[:,1]-cy))
            if d<bd: bd,best=d,li
        if best is not None and bd < 1.2*(wb[3]-wb[1]): groups[best].append(wb)
    boxes=[]
    for g in groups:
        if not g: continue
        g=np.array(g); boxes.append([float(g[:,0].min()),float(g[:,1].min()),float(g[:,2].max()),float(g[:,3].max()),1.0])
    out[fn]=boxes; print(fn,"lines",len(boxes),"gt",len(gt[fn]["lines"]),"words",len(words),"polylines",len(lines))
json.dump(out,open(ROOT/"eval/runs/rp_det_lines.json","w"))
