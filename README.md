Floorplan-recognition
==============================
------------------------------
## 1. Distinguishing floor plan image
### Structure:  
> #### (1) (input)64x64x3 ---> conv + pool  
> #### (2) 32x32x32 ---> conv + pool  
> #### (3) 16x16x64 ---> flatten + conv  
> #### (4) 1x1x800 ---> conv
> #### (5) 1x1x2(output)  
<div align = left><img width='700' height='160' src='https://github.com/Menglinucas/Floorplan-recognition/blob/master/RECG_CNN_structure.jpg'></div>  

### Train:  
> #### RECG_CNN.py train  
### Predict
> #### RECG_CNN.py predict
------------------------------
## 2. Extracting the truth part  
### Structure:  
> #### (1) (input)64x64x3 ---> conv + pool  
> #### (2) 32x32x32 ---> conv + pool  
> #### (3) 16x16x64 ---> transpose, stride = 2  
> #### (4-2) 32x32x32 ---> transpose, stride = 2  
> #### (5-1) 64x64x3(output)
### Train:  
> #### CUTOUT_FCN.py  
### Predict:  
> #### pred = sess.run(annotation_pred,feed_dict)  
> #### pred = np.squeeze(pred,axis=3)  
