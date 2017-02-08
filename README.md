README.md - explains the structure of your network and training approach.

(1) Structure of network
    a. The network is very similar to Nvidia's paper about end-to-end training of self driving car, having similar number of parameters.
    b. I made the following changes/additions to the original network
        (1) Use batch normalization before every activation(except the last two FC layers)
        (2) Use ELU intead of RELU
    c. Use Adam optimizer with default learning rate, thanks to batch normalization
    d. The mere regulization provided by batch normalization seems to be sufficient. Additionally, there are machanizms to drop out zero steering angles with certain probability so that the model avoid overfitting zero values. Please see the first point in "Training approach".

(2) Training approach
    a. One important step is to drop the images with close to zero steering angles with certain probability. This way, the model would tend to fit into those images with non-zero steering angles. The probability is controlled by a function parameter called "pr_threshold" by generate_from_directory. The idea is from this link: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9?source=user_profile---------6----------&gi=112f78fa9e29
    b. Train-test-split: Since the real "test" should be the car driving on the road, I did not allocate data for a dedicated test set. However, I do have 10% of the data set aside as validation set, so that I have indications whether my model has overfitting/underfitting problem.
    c. With the above drop-out strategies. I first tried to train the network with data provided by Udacity. Left-camera images are adjusted by adding 0.15 to steering angle accordingly. Right-camera images are adjusted by substracting 0.15 from steering angle. 
    d. After the above training (<5 epochs, default learning rate used), there are a few bad corners where the car would drive off the track
    e. Then I added a few thousands of recovery data. However, only center images are used from these recovery data. After 4 epochs of training, the car can drive within the safe zone of track #1.

(3) To-do (for my own record)
    a. Try use regular normalization (x-128/255.0 as the first layer). With drop-out methods described in "Training approach", does the model output a constant angles? If not, how is the learning speed compared with used batch normalization?
    b. Use other image augmentation(brightness, shift, flipping) to see if the model can generates to the 2nd track.
    c. Use PyCharm to read the batch normalization code