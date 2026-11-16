"""Render saved predictions on CPU; no model or simulator invocation."""
import argparse,json,subprocess
from pathlib import Path
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1152x570','-r','5','-i','-','-c:v','libx264','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)],stdin=subprocess.PIPE)
summary=[]
try:
 for file in sorted(a.episode.glob('forecast-*.npz')):
  data=np.load(file);anchor=int(file.stem.split('-')[-1]);truth=data['truth'];pred=data['prediction'];context=data['context']
  assert truth.shape==pred.shape==(20,128,128,3)
  mae=np.abs(pred-truth).mean(axis=(1,2,3))
  hold=np.abs(context[None].astype(np.float32)/255-truth).mean(axis=(1,2,3))
  summary.append({'anchor':anchor,'mae_at_2s':float(mae[-1]),'hold_mae_at_2s':float(hold[-1]),'mean_mae':float(mae.mean()),'mean_hold_mae':float(hold.mean())})
  for step in range(20):
   canvas=np.zeros((570,1152,3),np.uint8)
   actual=(np.clip(truth[step],0,1)*255).astype(np.uint8)
   prediction=(np.clip(pred[step],0,1)*255).astype(np.uint8)
   error=np.abs(pred[step]-truth[step]).mean(-1)
   error_color=cv2.applyColorMap((np.clip(error*4,0,1)*255).astype(np.uint8),cv2.COLORMAP_INFERNO)[:,:,::-1]
   for i,panel in enumerate([actual,prediction,error_color]):
    canvas[80:464,i*384:(i+1)*384]=cv2.resize(panel,(384,384),interpolation=cv2.INTER_NEAREST)
   for i,title in enumerate(['Actual future BEV','World-model prediction','Absolute error (4x)']):
    cv2.putText(canvas,title,(i*384+12,65),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
   cv2.putText(canvas,f'Context ends at {anchor/10:.1f}s | Forecast +{(step+1)/10:.1f}s | 0.5x playback',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
   cv2.putText(canvas,'Recorded future actions supplied. No future image is encoded.',(12,500),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
   cv2.putText(canvas,f'Pixel MAE {mae[step]:.4f} | Hold-current-frame baseline {hold[step]:.4f}',(12,530),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
   cv2.putText(canvas,'Native BEV 128x128, enlarged. Low image error does not establish safe driving.',(12,557),cv2.FONT_HERSHEY_SIMPLEX,.55,(200,200,200),1,cv2.LINE_AA)
   writer.stdin.write(canvas.tobytes())
finally:
 writer.stdin.close()
 assert writer.wait()==0
a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2))
