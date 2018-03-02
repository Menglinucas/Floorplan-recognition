Floorplan-recognition
==============================
## 1. Distinguishing floor plan image
------------------------------
### Training:  
> #### RECG_CNN.py train  
### Predict
> #### RECG_CNN.py predict
  
## 2. Extracting the truth part
------------------------------
### Training:  
> #### CUTOUT_FCN.py  
### prediction:  
> #### pred = sess.run(annotation_pred,feed_dict)  
> #### pred = np.squeeze(pred,axis=3)  
