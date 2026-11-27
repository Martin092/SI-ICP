import json
from pathlib import Path

import numpy as np
# from baselines import non_linear_noise, M_ICP, M_ICP_gaussian

from baselines import NonLinearNoise, SI_ICP, SI_ICP_Gaussian_Residuals, SI_ICP_Gaussian_Residuals_Fraction

class ExperimentLogger:
    '''
    Schema:
    {
        "meta": {
            "name": "...",
            "benchmark": "observational",
            "trials": 100,
            "size": [100, 1],
            "causal_model": {"name": "LinearAdditiveModel", "slope": 2, "intercept": 0}
        },
        "settings": [
            {
                "setting": 0,
                "x_distribution": "Normal(0, 1)",
                "noise_distribution": "Normal(0, 1)",
                "n_trials": 100,
                "success_rate": {"x->y": 0.95, "y->x": 0.62},
                "trials": [
                    {
                        "trial": 0,
                        "results": {
                            "x->y": {
                                "rejected_null": false,
                                "expected_reject": false,
                                "correct": true,
                                "p_null": 0.41,
                                "p_alt": 0.002,
                                "model_null": {"slope": [0.98], "intercept": 0.01},
                                "model_alt": null
                            },
                            "y->x": {...}
                        },
                        "data": {"x": [...], "y": [...], "noises": [...], "x_f": ..., "y_f": ...}
                    }
                ]
            }
        ]
    }
    '''
    def __init__(self, meta=None):
        self.meta = meta or {}
        self._settings = {}  # setting index -> {"info": {...}, "trials": [...]}

    def add(self, rec):
        s = self._settings.setdefault(rec["setting"], {
            "info": {
                "setting": rec["setting"],
                "x_distribution": rec["x_name"],
                "noise_distribution": rec["noise_name"],
            },
            "trials": [],
        })
        s["trials"].append({
            "trial": rec["trial"],
            "results": rec["results"],
            "data": {k: rec[k] for k in ("x", "y", "noises", "x_f", "y_f") if k in rec},
        })

    def to_dict(self):
        settings = []
        for s in self._settings.values():
            n = len(s["trials"])
            rates = {
                d: sum(t["results"][d]["correct"] for t in s["trials"]) / n
                for d in ("x->y", "y->x")
            }
            settings.append({**s["info"], "n_trials": n, "success_rate": rates, "trials": s["trials"]})
        return {"meta": self.meta, "settings": settings}

    def summary(self):
        return {
            f"{s['x_distribution']} | {s['noise_distribution']}": s["success_rate"]
            for s in self.to_dict()["settings"]
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=self._to_jsonable)

    @staticmethod
    def _to_jsonable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        return str(v)

    
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
        self.name = "LinearAdditiveModel"

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

        for i, (x_distr, noise_distr) in enumerate(zip(self.x_distributions, self.noise)):
            for k in range(self.trials):
                xs = x_distr.sample(self.size)
                noises = noise_distr.sample(self.size)
                ys = self.causal_model.compute(xs, noises)

                record = {
                    "setting": i,
                    "x_name": x_distr.name,
                    "noise_name": noise_distr.name,
                    "trial": k,
                    "x": xs,
                    "y": ys,
                    "noises": noises,
                }

                if self.benchmark_id == "interventional":
                    x_f, y_f = self._interventional_samples(i)
                    record["x_f"], record["y_f"] = x_f, y_f

                    alg_yx = self.method(ys, xs, y_f, x_f)
                    alg_xy = self.method(xs, ys, x_f, y_f)
                elif self.benchmark_id == "observational":
                    alg_yx = self.method(ys, xs)
                    alg_xy = self.method(xs, ys)
                else:
                    raise ValueError(f"Unknown benchmark_id: {self.benchmark_id}")

                record["results"] = {
                    "x->y": self._run_direction(alg_xy, expected_reject=False),
                    "y->x": self._run_direction(alg_yx, expected_reject=True),
                }
                yield record

    @staticmethod
    def _run_direction(alg, expected_reject):
        # ASSUMPTION: the first return value of alg.run() means "null was rejected".
        # If it means the opposite, change this one line.
        rejected_null, p_null, p_alt = alg.run()
        return {
            "rejected_null": bool(rejected_null),
            "expected_reject": expected_reject,
            "correct": bool(rejected_null) == expected_reject,
            "p_null": p_null,
            "p_alt": p_alt,
            # fits are logged in the frame the test ran in (no inversion)
            "model_null": Benchmark._fit_summary(alg.model_null),
            "model_alt": Benchmark._fit_summary(alg.model_alt),
        }

    @staticmethod
    def _fit_summary(model):
        if model is None:
            return None
        return {"slope": model.coef_, "intercept": model.intercept_}

    def run_and_log(self):
        logger = ExperimentLogger(meta={
            "name": self.name,
            "benchmark": self.benchmark_id,
            "trials": self.trials,
            "size": self.size,
            "causal_model": {
                "name": self.causal_model.name,
                "slope": self.causal_model.slope,
                "intercept": self.causal_model.intercept
            }
        })
        for record in self.run():
            logger.add(record)

        print(logger.summary())
        logger.save(Path("results") / self.name / "log.json")
        return logger.to_dict()


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

ObservationalBenchmark(
    distr,
    distr_noise, 
    causal_model,
    NonLinearNoise,
    name="non_gaussian_noise",
    trials=10
).run_and_log()


# InterventionalBenchmark(
#     distr, 
#     distr_noise,
#     causal_model, 
#     SI_ICP, 
#     interventional_distr,
#     interventional_noise,
#     name="SI_ICP",
#     trials=1000
# ).run_and_log()


# InterventionalBenchmark(
#     distr, 
#     distr_noise,
#     causal_model, 
#     SI_ICP_Gaussian_Residuals, 
#     interventional_distr,
#     interventional_noise,
#     name="SI_ICP_GAUSSIAN",
#     trials=1000
# ).run_and_log()

# InterventionalBenchmark(
#     distr, 
#     distr_noise,
#     causal_model, 
#     SI_ICP_Gaussian_Residuals_Fraction, 
#     interventional_distr,
#     interventional_noise,
#     name="SI_ICP_F_TEST",
#     trials=1000
# ).run_and_log()


