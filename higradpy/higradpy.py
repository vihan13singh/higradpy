import numpy as np
import pandas as pd
import scipy.stats

class HiGrad:
    
    def __init__(self, model = 'lm', nsteps = None,
                nsplits = 2, nthreads = 2,
                step_ratio = 1, n0 = np.nan, skip = 0,
                eta = 1/2, alpha = 1/2, burnin = None,
                start = None, replace = False, track = False):
        
        self.model = model;
        self.nsteps = nsteps;
        self.nsplits = nsplits;
        self.nthreads = nthreads;
        self.step_ratio = step_ratio;
        self.n0 = n0;
        self.skip = skip;
        self.eta = eta;
        self.alpha = alpha;
        self.burnin = burnin;
        self.start = start;
        self.replace = replace;
        self.track = track

    def fit(self, x, y):
        
        if self.model not in ['linear', 'logistic']:
            raise ValueError('Model Not Supported')
        if self.model == 'logistic':
            if len(np.unique(y)) != 2:
                raise ValueError("response is not a binary variable")
            else:
                y = pd.Series(y.reshape(1, -1)[0])
                y = ((y == max(y)).apply(int)*2 - 1).values

        if x.shape[0] < 2:
            raise ValueError("'x' should be a matrix with 2 or more rows?")
        if x.shape[0] != y.shape[0]:
            raise ValueError("the dimensions of 'x' and 'y' do not agree")

        # nsteps
        self.nsteps = x.shape[0]
        # burnin
        self.burnin = round(self.nsteps / 10)
        # start
        self.start = np.random.normal(0, 0.01, x.shape[1])

        # Create Split???
        self.split = self.createSplit(self.nsteps, self.nsplits, self.nthreads, self.step_ratio, self.n0)

        self.ns = self.split['ns']
        self.K = self.split['K']
        self.Bs = self.split['Bs']
        self.n0 = int(self.split['n0'])
        self.B = np.prod(self.Bs)
        self.ws = np.append(self.n0, self.ns) * np.cumprod(np.append(1, self.Bs))
        self.ws = self.ws / sum(self.ws)
        self.d = x.shape[1] # no. of columns

        # create gradient function
        if (self.model == "linear"):
            def getGradient(theta, x1, y1): return x1 * (sum(theta * x1) - y1)
        elif (self.model == "logistic"):
            def getGradient(theta, x1, y1): return -y1 * x1 / (1 + np.exp(y1 * sum(theta * x1)))

        self.idx = 0

        # fit higrad
        # burnin stage
        if self.burnin > 0:
            for i in range(1, self.burnin+1):
                
                self.idx = self.sampleNext(self.idx, x.shape[0], self.replace)
                
                self.start = self.start - self.stepSize(i) * getGradient(self.start, x[self.idx, ], y[self.idx])
                

        # create a matrix that stores stagewise average
        self.theta_avg = np.full((self.B, self.d, self.K+1), np.nan)
        

        # set up theta_track for plotting
        if self.track: 
            self.theta_track = np.full((self.d, 1), start.reshape(-1, 1))
            

        # zeroth stage
        # theta matrix for the current stage
        self.theta_current = np.full((int(self.d), int(self.n0+1)), np.nan)
        # initial value
        self.theta_current[:, 0] = self.start

        # iteration
        if self.n0 > 0:
            for i in range(0, self.n0):
                self.idx = self.sampleNext(self.idx, x.shape[0], self.replace)
                self.theta_current[:, i+1] = self.theta_current[:, i] - self.stepSize(i+1) * getGradient(self.theta_current[:, i], x[self.idx, :], y[self.idx])
            # average and store in theta_avg
            self.theta_avg[:, :, 0] = np.full((self.B, self.d), np.mean(self.theta_current[:, -np.array(range(0, int(np.floor(self.n0 * self.skip)+1)))], axis=1))       
        else:
            self.theta_avg[:, :, 0] = np.full((self.B, self.d), self.start)

        # concatenate theta_track
        if self.track:
            self.theta_track = np.append(self.theta_track, self.theta_current[:, :-1])

        # set initial value for the next stage
        self.start = np.full((self.Bs[0], self.d), self.theta_current[:, self.n0])

        for k in range(0, self.K):
            self.theta_current = np.full((int(np.cumprod(self.Bs)[k]), self.d, int(self.ns[k]+1)), np.nan) 
            self.theta_current[:,:,0] = self.start
            # iteration
            self.n_current = np.cumsum(np.append(self.n0, self.ns))[k]
            for i in range(0, int(self.ns[k])):
                for j in range(0, int(np.cumprod(self.Bs)[k])):
                    self.idx = self.sampleNext(self.idx, x.shape[0], self.replace)
                    self.theta_current[j, :, i+1] = self.theta_current[j, :, i] - self.stepSize(self.n_current+i) * getGradient(self.theta_current[j, :, i], x[self.idx-1, :], y[self.idx-1]) 
            # averaging        
            self.theta_avg[:, :, k+1] = np.repeat(np.mean(self.theta_current[:,:,np.array(range(0, int(np.floor(self.ns[k]*self.skip)+2)))[-1]:], axis=2).reshape(1, -1), int(self.B/np.cumprod(self.Bs)[k])).reshape(self.theta_avg[:,:,:].shape[1], self.theta_avg[:,:,:].shape[0]).T      
                        
            # set initial value for the next stage
            if k < self.K-1:
             #   start <- matrix(rep(theta.current[, , ns[k]+1], each = Bs[k+1]), cumprod(Bs)[k+1], d)  
                self.start = np.repeat(self.theta_current[:,:,int(self.ns[k])], self.Bs[k]).reshape(self.d, np.cumprod(self.Bs)[k+1]).T  
                
                

        # weighted average across stages
        #thetas <- t(sapply(1:B, function(i) theta_avg[i, , ] %*% ws))
        self.thetas = self.theta_avg@self.ws

        self.out = {}
        self.out['coefficients'] = np.mean(self.thetas, axis=0)
        self.out['coefficients_bootstrap'] = self.thetas
        self.out['model'] = self.model
        self.out['Sigma0'] = self.getSigma0(np.append(self.n0, self.ns), np.append(1, self.Bs), self.ws)
        self.out['track'] = np.nan
        if self.track:
            self.out['track'] = self.theta_track

    def predict(self, x, alpha=0.05, t="link", prediction_interval=False):
        
        obj = self.out

        # Dimension Check
    #    if (is.vector(newx)) {
     #   newx = matrix(newx, 1, length(newx))
      #  }
       # if (ncol(newx) != length(object$coefficients)) {
        #stop("'newx' has the wrong dimension")
        #}

        Sigma0 = obj['Sigma0']
        
        B = Sigma0.shape[0]
        
        mu = x@obj['coefficients']
        #print(mu)
    
        # n x B matrix of predicted values
        mus = x @ obj['coefficients_bootstrap'].T
    
        # standard errors
        ses = np.sqrt(sum(sum(Sigma0))*np.sum(
            (mus - mu.reshape(-1, 1)).T * np.linalg.solve(Sigma0, (mus - mu.reshape(-1, 1)).T),  axis=0) / (B**2 * (B-1)) )    
        
        if prediction_interval:
            margin = scipy.stats.t.ppf(1-alpha/2, B - 1) * ses * np.sqrt(2)
        else:
            margin = scipy.stats.t.ppf(1-alpha/2, B - 1) * ses 
        
        upper = mu + margin
        
        lower = mu - margin       

        if obj['model'] == "logistic" and t == "response":
            mu = 1 / (1 + np.exp(-mu))
            upper = 1 / (1 + np.exp(-upper))
            lower = 1 / (1 + np.exp(-lower))

        out = {}
        out['pred'] = mu
        out['upper'] = upper
        out['lower'] = lower
    
        return out

    def score(self, X, y):
        if self.model == "linear":
            from sklearn.metrics import regression
            y_hat = self.predict(X)['pred']
            y_hat = y_hat.reshape(-1, 1)
            return regression.mean_squared_error(y, y_hat)
        elif self.model == "logistic":
            y = pd.Series(y.reshape(1, -1)[0])
            y = ((y == max(y)).apply(int)*2 - 1).values
            from sklearn.metrics import classification
            y_hat = self.predict(X, t='response')['pred']
            y_hat = np.where(y_hat > 0.5, 1, -1)
            return classification.accuracy_score(y, y_hat)    
    
    def getSigma0(self, ns, Bs, ws=None):
        B = np.prod(Bs)
        BBs = np.cumprod(Bs)
        if ws is None:
            ws = ns * BBs
            ws = ws / sum(ws)
        Sigma0 = np.zeros((B, B))
        for l in range(np.where(ns > 0)[0][0], len(ns)):
            Sigma0 = Sigma0 + np.kron(np.eye(BBs[l],dtype=int),np.full((int(B/BBs[l]), int(B/BBs[l])), ws[l]**2/ns[l]))
        return Sigma0

    # create configurations based on the input parameters
    def createSplit(self, nsteps, nsplits, nthreads, step_ratio, n0):

        N = nsteps
        
        K = nsplits
        
        nthreads = np.array([nthreads])
        if len(nthreads) == 1:
            Bs = np.repeat(nthreads, K, axis=0)
        else:
            Bs = nthreads

        if len(Bs) != K:
            raise CustomError

        step_ratio = np.array(step_ratio)**range(0, K)
        
        if np.isnan(n0):
            ns = ((N / sum(np.append(1, step_ratio) * np.append(1, np.cumprod(Bs)))) * np.append(1, step_ratio)).round()   
            n0 = ns[0]
            ns = ns[1:]
        else:
            ns = np.round(((N - n0) / sum(step_ratio * np.cumprod(Bs))) * step_ratio)

        ns = ns[ns > 0]
        L = sum(ns > 0)
        Bs = Bs[ns > 0]

        return {"ns":ns, "K":L, "Bs":Bs, "n0":n0}
    
    # create step function
    def stepSize(self, n):
        
        return self.eta / n**self.alpha

    # create sample function
    def sampleNext(self, current, n, replace):
        
        if replace: 
            return np.random.choice(n, 1)
        else: 
            return current % n + 1

