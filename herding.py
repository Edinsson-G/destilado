import numpy as np
from scipy.optimize import minimize #to do grad descent to fing argmax
def expKer(x,samples,gamma):
    #--calcuates the expectation value of the exponential kernel so argmax_x can be found
    #x = candidate super sample to optimize
    #samples = the GMM samples
    #gamma = kernel hyperparameter, always 1 for my demo
    
    #init vars
    numSamples = samples.shape[0]
    k=np.zeros(numSamples)
    #calculate estimate of expectation value of kernel
    for i in range(numSamples):
        k[i] = np.exp(-np.linalg.norm(x-samples[i,:])/gamma**2)
    exp_est = sum(k)/numSamples;
    return exp_est

def sumKer(x,xss,numSSsoFar,gamma):
    #--calcuates the sum of k(x,x_ss) for the number of super samps so far
    #x = candidate super sample to optimize
    #samples = the GMM samples
    #numSSsoFar = number of super sampls so far
    #gamma = kernel hyperparameter, always 1 for my demo
    
    #init vars
    total=0;
    k=np.zeros(numSSsoFar)
    #calculate sof of kernels
    for i in range(numSSsoFar):
        k[i] = np.exp(-np.linalg.norm(x-xss[i,:])/gamma**2)
    total = np.sum(k)
    s = total/(numSSsoFar+1)
    return s
def herd(samples,totalSS,gamma):
    #-- calculate totalSS super samples from the distribution estimated by samples with kernel hyperparam gamma
    
    #init vars and extract useful info from inputs
    #get GMM dims and num samples
    numDim = samples.shape[1]
    numSamples = samples.shape[0]
    
    #init vars
    gradientFail = 0; #count when optimization fails, debugging
    xss = np.zeros((totalSS,numDim)) #open space in mem for array of super samples
    i=1
    #gradient descent can have some probems, so make bounds to terminate if goes too far away
    minBound = np.min(samples)
    maxBound = np.max(samples)
    #start our search at the origin, could be a random point
    bestSeed = np.zeros(numDim)
    
#     tick = time.clock()
    while i<totalSS:
        #debugging stuff
        #print "Working on SS num ber i=%d" % i
        #build function for gradient descent to find best point
        f = lambda x: -expKer(x,samples,gamma)+sumKer(x,xss,i,gamma)
        results = minimize(f,
                           bestSeed,
                           method='nelder-mead',
                           options={'xtol': 1e-4, 'disp': False})
#         print "results.x"
#         print results.x
        
        #if grad descent failed, pick a random sample and try again
        if np.min(results.x) < minBound or np.max(results.x) > maxBound:
            bestSeed=samples[np.random.choice(numSamples)]
            gradientFail=gradientFail+1
#             print "Gradient descent failed.............."
            continue
        
        #pick next best start point to start minimization, this is how Chen, Welling, Smola do it
        #find best super sample that maximizes argmax and use that as a seed for the next search
        #init or clear seed array
        seed=np.array([])
        for j in range(i):
            seed = np.append(seed,-expKer(xss[j,:],samples,gamma)+sumKer(xss[j,:],xss,i,gamma))
        bestSeedIdx = np.argmin(seed)
        bestSeed=xss[bestSeedIdx,:]
        
        #grad descent succeeded (yay!), so assign new value to super samples
        xss[i,:]=results.x
        
        i=i+1
    return xss