# SPIGAN    

# discription  
 Implementation of SPIGAN using tensorflow.  
 
# official implementation  
I cannot find official implementation yet......

# literature  
 [SPIGAN](https://arxiv.org/abs/1810.03756)  

# dependency  
I confirmed operation only with..   
1)python 3.6.3  
2)tensorflow 1.7.0  
3)numpy 1.14.2    
4)Pillow 4.3.0  

# TODO  
1. metrics  
2. test phase  

# result images  
### result for training data  
This is the output from the task predictor of training data after 20 epochs.  

![trainResultImage_19042102_bc16_bcp16_19](https://user-images.githubusercontent.com/15444879/56515460-ebe76b80-6572-11e9-88d9-7d223a9eef64.png)

From left to right, synthesis image, adapted image (output from generator), output from task predictor of synthesis image, output from task predictor of adapted image, ground truth.   

### result for test CityScape data  
This is the output from the task predictor of test data after 20 epochs.  

![RealResultImage_19042102_bc16_bcp16_19](https://user-images.githubusercontent.com/15444879/56515448-e25e0380-6572-11e9-8bc4-0d737b70aa7e.png)

From left to right, test CityScape image, output from task predictor of it, ground truth.   

# email  
t.ohmasa@w-farmer.com  
