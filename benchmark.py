import json
from pathlib import Path

import numpy as np
# from baselines import non_linear_noise, M_ICP, M_ICP_gaussian

from baselines import NonLinearNoise, SI_ICP, SI_ICP_Gaussian_Residuals, SI_ICP_Gaussian_Residuals_Fraction

class Noise():
    def __init__(self):
        self.name = "UNNAMED NOISE DISTRIBUTION_" + str(np.random.rand())

    def sample(self, size):
        pass

class UniformNoise(Noise):
    def __init__(self, a, b):
        super().__init__()

        self.name = f"Uniform({a}, {b})"
        self.a = a 
        self.b = b 

    def sample(self, size=1):
        return np.random.uniform(self.a, self.b, size=size)

class TDistr(Noise):
    def __init__(self, dof):
        super().__init__()

        self.name = f"T({dof})"
        self.dof = dof 

    def sample(self, size=1):
        return np.random.standard_t(self.dof, size=size)

class Normal(Noise):
    def __init__(self, mu, var):
        super().__init__() 

        self.name = f"Normal({mu}, {var})"
        self.mu = mu 
        self.var = var  

    def sample(self, size=1):
        return np.random.normal(self.mu, self.var, size=size)

class Point(Noise):
    def __init__(self, x):
        super().__init__() 

        self.name = f"Point({x})"
        self.x = x 

    def sample(self, size=1):
        return np.ones(size) * self.x


class LinearAdditiveModel():
    def __init__(self, slope, intercept):
        self.slope = slope 
        self.intercept = intercept 

    def compute(self, xs, noises):
        return self.slope * xs + self.intercept + noises

    def json(self):
        return {
            "type": "LinearAdditiveModel",
            "slope": self.slope,
            "intercept": self.intercept
        }

class Benchmark:
    def __init__(
            self, 
            x_distributions, 
            noise_distributions, 
            causal_model, 
            method, 
            size=(100, 1), 
            trials=10,
            name = np.random.rand()
    ):

        self.x_distributions = x_distributions
        self.noise = noise_distributions
        self.causal_model = causal_model
        self.method = method   
        self.size = size
        self.trials = trials

        self.benchmark_id = None
        self.name = str(name)

        self.log = dict() 
        self.log["causal_model"] = self.causal_model.json()
        self.log["trials"] = dict()

    def _interventional_samples(self, i):
        return None

    def run(self):
        assert len(self.x_distributions) == len(self.noise)

        
        for i in range(len(self.x_distributions)):
            x_distr = self.x_distributions[i]
            noise_distr = self.noise[i]
            
            key_trial = f"{x_distr.name} {noise_distr.name} success"
            self.log['trials'][key_trial] = 0

            key_data = f"{x_distr.name} {noise_distr.name} data"
            self.log[key_data] = dict() 
            self.log[key_data]['distr_name'] = f"{x_distr.name}; {noise_distr.name}"
            
            for k in range(self.trials):
                xs = x_distr.sample(self.size) 
                noises = noise_distr.sample(self.size) 

                ys = self.causal_model.compute(xs, noises) 

                self.log[key_data][k] = dict()
                self.log[key_data][k]["x"] = xs 
                self.log[key_data][k]["y"] = ys 
                self.log[key_data][k]["noises"] = noises 

                discovery_alg = None 
                if self.benchmark_id == "interventional":
                    x_f, y_f = self._interventional_samples(i)

                    self.log[key_data][k]["x_f"] = x_f 
                    self.log[key_data][k]["y_f"] = y_f

                    discovery_alg = self.method(xs, ys, x_f, y_f)
                elif self.benchmark_id == "observational":
                    discovery_alg = self.method(xs, ys)

                is_correct, pval_x_to_y, pval_y_to_x = discovery_alg.run()

                self.log[key_data][k]['pval_x_to_y'] = pval_x_to_y 
                self.log[key_data][k]['pval_y_to_x'] = pval_y_to_x 

                if discovery_alg.model_x_to_y is not None:
                    self.log[key_data][k]['x_to_y_model'] = {
                        "slope": discovery_alg.model_x_to_y.coef_,
                        "intercept": discovery_alg.model_x_to_y.intercept_
                    }

                if discovery_alg.model_y_to_x is not None:
                    self.log[key_data][k]['y_to_x_model'] = {
                        "slope": discovery_alg.model_y_to_x.coef_,
                        "intercept": discovery_alg.model_y_to_x.intercept_
                    }
                
                self.log[key_data][k]["correct"] = is_correct

                if is_correct: self.log['trials'][key_trial] += 1

            self.log['trials'][key_trial] /= self.trials

        print(self.log['trials'])

        results_dir = Path("results") / self.name
        results_dir.mkdir(parents=True, exist_ok=True)
        with (results_dir / "log.json").open("w", encoding="utf-8") as log_file:
            json.dump(
                self.log,
                log_file,
                indent=2,
                default=lambda value: value.tolist() if isinstance(value, np.ndarray)
                else value.item() if isinstance(value, np.generic)
                else str(value),
            )


class ObservationalBenchmark(Benchmark):
    def __init__(self, x_distributions, noise_distributions, causal_model, method, size=(100, 1), trials=10, name=np.random.rand()):
        super().__init__(x_distributions, noise_distributions, causal_model, method, size, trials, name)

        self.benchmark_id = "observational"

    

class InterventionalBenchmark(Benchmark):
    def __init__(
            self,
            x_distributions,
            noise_distributions,
            causal_model,
            method,
            interventional_x, 
            interventional_noise,
            size=(100, 1),
            trials=10,
            name=np.random.rand()
    ):
        super().__init__(x_distributions, noise_distributions, causal_model, method, size=size, trials=trials, name=name)
        self.interventional_x = interventional_x
        self.interventional_noise = interventional_noise

        self.benchmark_id = "interventional"

    def _interventional_samples(self, i):
        # print(i) 
        x_f = self.interventional_x[i].sample()
        noise = self.interventional_noise[i].sample()

        y_f = self.causal_model.compute(x_f, noise)

        return x_f, y_f





distr = [
    UniformNoise(-1, 1),
    TDistr(3),
    TDistr(3),
    TDistr(10),
    Normal(0, 1),
]

distr_noise = [
    UniformNoise(-1, 1),
    UniformNoise(0, 1),
    TDistr(3),
    TDistr(10),
    Normal(0, 1)
]

causal_model = LinearAdditiveModel(2, 0)



interventional_distr = [Point(5)] * len(distr_noise)
interventional_noise = [Point(0)] * len(distr_noise)

# ObservationalBenchmark(
#     distr,
#     distr_noise, 
#     causal_model,
#     NonLinearNoise,
#     name="non_linear",
#     trials=1000
# ).run()

# InterventionalBenchmark(
#     distr, 
#     distr_noise,
#     causal_model, 
#     SI_ICP, 
#     interventional_distr,
#     interventional_noise,
#     name="SI_ICP",
#     trials=1000
# ).run()


# InterventionalBenchmark(
#     distr, 
#     distr_noise,
#     causal_model, 
#     SI_ICP_Gaussian_Residuals, 
#     interventional_distr,
#     interventional_noise,
#     name="SI_ICP_GAUSSIAN",
#     trials=1000
# ).run()

InterventionalBenchmark(
    distr, 
    distr_noise,
    causal_model, 
    SI_ICP_Gaussian_Residuals_Fraction, 
    interventional_distr,
    interventional_noise,
    name="SI_ICP_F_TEST",
    trials=1000
).run()


