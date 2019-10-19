import scipy.io
import numpy as np


matlabeltrain0 = scipy.io.loadmat('mat/train_0_label.mat')
matlabeltrain1 = scipy.io.loadmat('mat/train_1_label.mat')
matimgtrain0 = scipy.io.loadmat('mat/train_0_img.mat')
matimgtrain1 = scipy.io.loadmat('mat/train_1_img.mat')
matlabeltest0 = scipy.io.loadmat('mat/test_0_label.mat')
matlabletest1 = scipy.io.loadmat('mat/test_1_label.mat')
mattestimg0 = scipy.io.loadmat('mat/test_0_img.mat')
mattestimg1 = scipy.io.loadmat('mat/test_1_img.mat')


#Calculating average brightness(feature1) of the image
def get_avg_brightnees(imgarry):
    x = np.array(imgarry['target_img'])
    y=x.transpose()
    avg_bright=[]
    for img in y:
        sum=float(0)
        for img_r in img:
            for bright in img_r:
                 sum=sum+bright
        avg=float(sum)/784
        avg_bright.append(avg)
    return avg_bright

#Calculating the average of the variances of each rows (feature2) of the image.
def get_avg_variance(imgarry):
    x = np.array(imgarry['target_img'])
    y=x.transpose()
    
    avg_var=[]
    for img in y:
        sum_var = 0
        for img_r in img:
            sum_row=0
            var_sum = float(0)
            for bright in img_r:
                sum_row=sum_row+bright
            mean_row=float(sum_row)/28
            for bright in img_r:
                var_sum=var_sum+((bright-mean_row)*(bright-mean_row))
            var_row = var_sum /28
            sum_var = sum_var + var_row
        avg_var_img = sum_var / 28
        avg_var.append(avg_var_img)
    return avg_var



# Estimate mean(mu) parameter of normal distribution using MLE density estimation algo 
def get_mlemu(muarry):
    mu=0
    mumean=0
    for i in muarry:
        mu=mu+i
    mumean=float(mu)/len(muarry)
    return mumean

# Estimate variance(sigma2) parameter of normal distribution using MLE density estimation algo 
def get_mlesigma2(sigmarry):
    sigma=0
    sigmamean=0
    sigmavar=0
    sigmavalue=0
    for i in sigmarry:
        sigma=sigma+i
    sigmamean=float(sigma)/len(sigmarry)
    for i in sigmarry:
        sigmavar=sigmavar+((i-sigmamean)*(i-sigmamean))
    sigmavalue=float(sigmavar)/len(sigmarry)
    return sigmavalue
    

# Function that calculates likilihood p(x | y):
def p_x_given_y(x, mean_y, variance_y):
    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
    return p

# Prior probality 
p__prior_label_0= 0.5
p__prior_label_1=0.5


#  Return the accuracy given the features of testing data  
#  Explanation: 1. Calculate likelihood for both the features feature 1 and feature 2.
#               2. Calculate the posterior odds from the prior and likelihood of feature1 and feature2.
#               3. Do the above steps for label 0 and label 1 and compare them.
#               4. If the posterior probability for label 0 is greater than that of label 1, 
#                 we recognize the image as the label 0. Similarly, If the posterior probability 
#                 for label 1 is greater than that of label 0, we recognize the image as the label 1.   
def get_accuracy(feature1_test, feature2_test, label_class ):
    predicted_count_0 = 0
    predicted_count_1 = 0
    
    for i in range(len(feature1_test)):
        likelihood_feature1_label_0 = p_x_given_y(feature1_test[i], mean_feature1_train_0, variance_feature1_train_0) 
        likelihood_feature2_label_0 = p_x_given_y(feature2_test[i], mean_feature2_train_0, variance_feature2_train_0)
        #Using the Bayesian formula to find posterior
        posterior_0 = p__prior_label_0 * likelihood_feature1_label_0 * likelihood_feature2_label_0

        likelihood_feature1_label_1 = p_x_given_y(feature1_test[i], mean_feature1_train_1, variance_feature1_train_1) 
        likelihood_feature2_label_1 = p_x_given_y(feature2_test[i], mean_feature2_train_1, variance_feature2_train_1)
        #Using the Bayesian formula to find posterior
        posterior_1 = p__prior_label_1 * likelihood_feature1_label_1 * likelihood_feature2_label_1

        if(posterior_0 > posterior_1):
            predicted_count_0 = predicted_count_0 + 1
        else:
            predicted_count_1 = predicted_count_1 + 1
            
    if(label_class == 0):
        print "\nPredicting the images from test class(label) 0 ..."
        predicted_count = predicted_count_0
    else:
        print "\nPredicting the images from test class(label) 1 ..."
        predicted_count = predicted_count_1
                
    print "Total number of images that are predicted as 0:", predicted_count_0 
    print "Total number of images that are predicted as 1:", predicted_count_1
        
    accuracy_label = predicted_count *float(100)/len(feature1_test)
    
    return accuracy_label

print("Calculating feature 1 and feature 2 for training data..")
#Extrating the feature 1 and feature 2 for the training data      
feature1_train_0=get_avg_brightnees(matimgtrain0)
feature1_train_1=get_avg_brightnees(matimgtrain1)
feature2_train_0=get_avg_variance(matimgtrain0)
feature2_train_1=get_avg_variance(matimgtrain1)

print("Calculating feature 1 and feature 2 for testing data..")
#Extrating the feature 1 and feature 2 for the testing data
feature1_test_0=get_avg_brightnees(mattestimg0)
feature1_test_1=get_avg_brightnees(mattestimg1)
feature2_test_0=get_avg_variance(mattestimg0)
feature2_test_1=get_avg_variance(mattestimg1)

# Estimate the parameters using MLE Density Estimation from the training data for feature 1
mean_feature1_train_0=get_mlemu(feature1_train_0)
print "\nMean parameter of the feature 1 for label 0: ",mean_feature1_train_0
mean_feature1_train_1=get_mlemu(feature1_train_1)
print "Mean parameter of the feature 1 for label 1: ",mean_feature1_train_1
variance_feature1_train_0=get_mlesigma2(feature1_train_0)
print "Variance parameter of the feature 1 for label 0: ", variance_feature1_train_0
variance_feature1_train_1=get_mlesigma2(feature1_train_1)
print "Variance parameter of the feature 1 for label 1: ", variance_feature1_train_1

# Estimate the parameters using MLE Density Estimation from the training data for feature 2
mean_feature2_train_0=get_mlemu(feature2_train_0)
print "\nMean parameter of the feature 2 for label 0: " , mean_feature2_train_0 
mean_feature2_train_1=get_mlemu(feature2_train_1)
print "Mean parameter of the feature 2 for label 1: ", mean_feature2_train_1 
variance_feature2_train_0=get_mlesigma2(feature2_train_0)
print "Variance parameter of the feature 2 for label 0: " , variance_feature2_train_0
variance_feature2_train_1=get_mlesigma2(feature2_train_1)
print "Variance parameter of the feature 2 for label 1: ", variance_feature2_train_1

#calculating  accuracy of predicting label
accuracy_test_0=get_accuracy(feature1_test_0, feature2_test_0, 0 )
accuracy_test_1=get_accuracy(feature1_test_1, feature2_test_1, 1 )
print "\n################# Accuracy #################\n"
print "Accuracy of predicting label 0:", accuracy_test_0
print "Accuracy of predicting label 1:", accuracy_test_1
    
print "\n################# End #####################\n"

    




