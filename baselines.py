import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import f
from hyppo.independence import Hsic

def normal_pdf(x, mu, var):
    Z = 1 / np.sqrt(2 * np.pi * var)
    e = np.exp(-((x - mu)**2) / (2 * var))
    return Z * e

def sample_in_distribution_test(ref, x, alpha=0.05):
    """H0: x is drawn from the same distribution as ref (exchangeable with it).
    Two-sided empirical tail test. Returns (p_value, reject_H0)."""
    ref = np.asarray(ref, float).ravel()
    n = len(ref)

    # +1 counts x itself, which makes the p-value exactly valid under H0
    p_low  = (1 + np.sum(ref <= x)) / (n + 1)
    p_high = (1 + np.sum(ref >= x)) / (n + 1)
    p = min(1.0, 2 * min(p_low, p_high))
    return p, p < alpha


class CausalDiscoveryObservational:
    def __init__(self, X, Y, significance_level=0.1):
        self.X = X 
        self.Y = Y

        self.model_x_to_y = None 
        self.model_y_to_x = None 

        self.significance_level = significance_level

        self.name = "No Name"

    def run():
        pass 

class NonLinearNoise(CausalDiscoveryObservational):
    def __init__(self, X, Y, significance_level=0.1):
        super().__init__(X, Y, significance_level)
        
        self.name = "Non-Linear Noise"

    def run(self):
        x_to_y = LinearRegression()
        y_to_x = LinearRegression()

        self.model_x_to_y = x_to_y 
        self.model_y_to_x = y_to_x

        X = self.X 
        Y = self.Y 
    
        x_to_y.fit(X, Y)
        y_to_x.fit(Y, X)
    
        res_x_to_y = x_to_y.predict(X) - Y
        res_y_to_x = y_to_x.predict(Y) - X  
    
        stat, pval_x_to_y = Hsic().test(res_x_to_y, X)
        stat, pval_y_to_x = Hsic().test(res_y_to_x, Y)

        return pval_x_to_y > pval_y_to_x, pval_x_to_y, pval_y_to_x


class CausalDiscoveryInterventional(CausalDiscoveryObservational):
    def __init__(self, X, Y, faulty_x, faulty_y, significance_level=0.1):
        super().__init__(X, Y, significance_level)

        self.faulty_x = faulty_x 
        self.faulty_y = faulty_y     

class SI_ICP(CausalDiscoveryInterventional):
    def __init__(self, X, Y, faulty_x, faulty_y, significance_level=0.1):
        super().__init__(X, Y, faulty_x, faulty_y, significance_level)   

    def _one_direction(self, x_to_y):
        X = self.X 
        Y = self.Y 
        faulty_x = self.faulty_x 
        faulty_y = self.faulty_y
        if not x_to_y:
            X = self.Y
            Y = self.X
            faulty_x = self.faulty_y
            faulty_y = self.faulty_x

        # Null: X -> Y
        model=LinearRegression() 
        model.fit(X, Y) # We fit the model in the correct direction X->Y
        residuals_normal=(Y-model.predict(X))**2
        residuals_faulty=(faulty_y-model.predict(faulty_x.reshape(1, -1)))**2

        pval = 1-(residuals_faulty > residuals_normal).mean()

        if x_to_y:
            self.model_x_to_y = model 
        else: 
            self.model_y_to_x = model 

        return pval

    def run(self):
        pval_x_to_y = self._one_direction(x_to_y=True)
        pval_y_to_x = self._one_direction(x_to_y=False)
    
        return pval_x_to_y > pval_y_to_x, pval_x_to_y, pval_y_to_x

class SI_ICP_Gaussian_Residuals(CausalDiscoveryInterventional):
    def __init__(self, X, Y, faulty_x, faulty_y, significance_level=0.1):
        super().__init__(X, Y, faulty_x, faulty_y, significance_level) 

    def _one_direction(self, x_to_y):
        X = self.X 
        Y = self.Y 
        faulty_x = self.faulty_x 
        faulty_y = self.faulty_y
        if not x_to_y:
            X = self.Y
            Y = self.X
            faulty_x = self.faulty_y
            faulty_y = self.faulty_x

        model=LinearRegression() 
        model.fit(X, Y) # We fit the model in the correct direction X->Y
        residuals_normal=(Y-model.predict(X))

        mean = residuals_normal.mean()
        var = residuals_normal.var()

        faulty_residual = (faulty_y - model.predict(faulty_x.reshape(1, -1)) )

        pval = normal_pdf(faulty_residual.flatten()[0], mean, var)

        if x_to_y:
            self.model_x_to_y = model 
        else: 
            self.model_y_to_x = model

        return pval

    def run(self):
        pval_x_to_y = self._one_direction(x_to_y=True)
        pval_y_to_x = self._one_direction(x_to_y=False)


        return pval_x_to_y > pval_y_to_x, pval_x_to_y, pval_y_to_x
    

class SI_ICP_Gaussian_Residuals_Fraction(CausalDiscoveryInterventional):
    def __init__(self, X, Y, faulty_x, faulty_y, significance_level=0.1):
        super().__init__(X, Y, faulty_x, faulty_y, significance_level) 

    def run(self):
        model=LinearRegression() 
        model.fit(self.X, self.Y) # We fit the model in the correct direction X->Y
        residuals_normal=(self.Y-model.predict(self.X))**2

        faulty_residual = (self.faulty_y - model.predict(self.faulty_x.reshape(1, -1))) ** 2

        self.model_x_to_y = model 

        statistic = faulty_residual / residuals_normal.mean() 
        statistic = statistic.flatten()[0]

        p_value = f.sf(statistic, 1, len(residuals_normal))

        return p_value > self.significance_level, p_value, -1