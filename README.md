# Brain-MRI-Segmentation

# Problem
The task is classify the image i.e. have tumor or not.Then if there is a tumor, segment that part of tumor

# Approach
For classification used the transfer learning model and for segmentation used the ResUNet Architecture

# Results
Used callbacks to avoid overfitting.For classification task training acuracy achieved is 96.92% and the testing accuracy achieved is 93.44% and defined the custom loss function i.e. **Tversky Loss** and achieved 94.59% training tversky accuracy  and 86.26% testing tversky accuracy.


### Classification Scores
![Screenshot 2022-07-04 at 5 55 45 PM](https://user-images.githubusercontent.com/75982871/179355460-efc278ff-cb8b-4b75-9ca6-5ce2ceaeacf3.png)

 
### Segmentation Scores
![Screenshot 2022-07-04 at 5 56 08 PM](https://user-images.githubusercontent.com/75982871/179355509-fab143f7-2f2e-4084-90b1-2e7e6f8e1879.png)


# Flask Demo
Made a flask app for testing the model that can detect the tumor. Below is the demo of the web app 

https://user-images.githubusercontent.com/75982871/179355426-ced459e4-38d3-4d8f-9ddc-9e99ccc734f1.mov

<br>

# Predictions 
Here are the below predictions for 4 images

![Screenshot 2022-07-04 at 5 58 32 PM](https://user-images.githubusercontent.com/75982871/179355617-5cbda512-9106-4478-81a7-98f023ef9248.png)


