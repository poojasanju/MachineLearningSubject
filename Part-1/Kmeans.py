import scipy.io
import numpy as np


matlabeltrain = scipy.io.loadmat('mat/KSamples.mat')

a= matlabeltrain['a']
print "the value", a
