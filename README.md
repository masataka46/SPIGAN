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

![trainResultImage_19042401_bc64_bcp64_resta5_17](https://user-images.githubusercontent.com/15444879/56709368-dc774680-675b-11e9-85cb-a0f8e684a692.png)  

From left to right, synthesis image, adapted image (output from generator), output from task predictor of synthesis image, output from task predictor of adapted image, ground truth.   

### result for test CityScape data  
This is the output from the task predictor of test data after 20 epochs.  

![RealResultImage_19042401_bc64_bcp64_resta5_17](https://user-images.githubusercontent.com/15444879/56709377-e731db80-675b-11e9-97fb-9439ae89070d.png)  

From left to right, test CityScape image, output from task predictor of it, ground truth.   

# email  
t.ohmasa@w-farmer.com  
