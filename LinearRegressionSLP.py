from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from enum import Enum
from scipy.stats import shapiro, bartlett
from math import inf
from tabulate import tabulate
from scipy.stats import t
from functools import lru_cache


class RegressionType(Enum):
    Linear = 1
    Ridge = 2
    Lasso = 3


class RegressionUtility:
    def __init__(
        self,
        device,
        x,
        y,
        epochs,
        variable_names=None,
        regression_type=RegressionType.Linear,
        learning_rate=0.01,
        delta_to_early_stop=1e-9,
        patience=3,
        penalty=None,
        cache_loss=False,
    ):
        (
            self.device,
            self.learning_rate,
            self.epochs,
            self.delta_to_early_stop,
            self.patience,
        ) = device, learning_rate, epochs, delta_to_early_stop, patience
        print(
            "This is a single layer perceptron based implementation that already has a bias term. Please do not append a vector of ones to design matrix X to estimate"
        )
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        if (variable_names is not None) and (len(variable_names) != x.shape[1]):
            raise ValueError(
                "Number of variable names should be equal to the number of columns in the design matrix"
            )
        if variable_names is None:
            self.variable_names = [f"Variable {i}" for i in range(x.shape[1])]
        else:
            self.variable_names = variable_names
        self.regression_type = regression_type
        self.penalty = penalty
        self.model = self._build_model().to(device)
        self.y_numpy = self.y.detach().cpu().numpy()
        self.x_numpy = self.x.detach().cpu().numpy()
        self.cache_loss = cache_loss
        self.intercept = None

    @property
    def degrees_of_freedom(self):
        x = self.design_matrix_with_intercept
        return x.shape[0] - x.shape[1]

    @property
    def design_matrix_with_intercept(self):
        ones_vector = torch.ones(self.x.size()[0], 1).to(self.device)
        return torch.cat((self.x, ones_vector), axis=1).to(self.device)

    @lru_cache(maxsize=None)
    def _get_t_statistic(self, coefficient, standard_error):
        return coefficient / standard_error

    @lru_cache(maxsize=None)
    def _get_p_value(self, coefficient, standard_error):
        t_statistic = self._get_t_statistic(coefficient, standard_error)
        return 2 * (1 - t.cdf(abs(t_statistic), self.degrees_of_freedom))

    def coefficients_and_pvalue(self):
        standard_error = self.standard_error_of_coefficients.flatten()
        self.coefficients = [
            (
                "Intercept",
                self.intercept[0],
                standard_error[-1],
                self._get_t_statistic(self.intercept[0], standard_error[-1]),
                self._get_p_value(self.intercept[0], standard_error[-1]),
            )
        ]
        self.coefficients.extend(
            [
                (
                    name,
                    number,
                    standard_error,
                    self._get_t_statistic(number, standard_error),
                    self._get_p_value(number, standard_error),
                )
                for name, number, standard_error in zip(
                    self.variable_names,
                    self.coefficients_without_intercept.flatten().tolist(),
                    standard_error[:-1],
                )
            ]
        )
        return self.coefficients

    def _pretty_print_coefficients(self):
        coefficients = self.coefficients_and_pvalue()

        return (
            tabulate(
                coefficients,
                headers=[
                    "Variable",
                    "Coefficient",
                    "Standard Error",
                    "t-Stat",
                    "P-value",
                ],
                tablefmt="psql",
            )
            + "\n\n"
        )

    def _intercept(self):
        ones_vector = torch.ones(self.x.size()[0], 1).to(self.device)
        self.x = torch.cat((self.x, ones_vector), axis=1).to(self.device)

    def _build_model(self):
        if self.regression_type == RegressionType.Linear:
            return nn.Linear(self.x.shape[1], 1)
        elif self.regression_type == RegressionType.Ridge:
            return nn.Linear(self.x.shape[1], 1)
        elif self.regression_type == RegressionType.Lasso:
            return nn.Linear(self.x.shape[1], 1)
        else:
            raise ValueError("Invalid regression type")

    def fit(self):
        patience = 0
        if self.cache_loss:
            self.loss_cache = []
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.regression_type == RegressionType.Linear:
            iteration = tqdm(
                range(self.epochs), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
            )
            loss_history, indexer = [0, inf], 0
            for epoch in iteration:
                optimizer.zero_grad()
                outputs = self.model(self.x)
                loss = criterion(outputs, self.y.view(-1, 1))
                loss.backward()
                optimizer.step()
                iteration.set_description(f"Loss: {loss.item()}")
                if self.cache_loss:
                    self.loss_cache.append(loss.item())
                if epoch % 100 == 0:
                    loss_history[indexer] = loss.item()
                    indexer = int(not indexer)
                    delta = abs(loss_history[0] - loss_history[1])
                    if delta < self.delta_to_early_stop:
                        patience += 1
                        print(
                            f"Early stop signal at epoch {epoch} since delta loss {delta} is less than {self.delta_to_early_stop}"
                        )
                        print(
                            f"Will be patient for {self.patience - patience} more times"
                        )
                        if patience > self.patience:
                            break

        elif self.regression_type == RegressionType.Ridge:
            # Implement Ridge regression here
            pass
        elif self.regression_type == RegressionType.Lasso:
            # Implement Lasso regression here
            pass

        self.intercept = self.model.bias.detach().cpu().numpy()
        # self.y = self.y.detach().cpu().numpy()
        # Compute R-squared and adjusted R-squared
        # self.r_squared = self._compute_r_squared()

    @property
    def r_squared(self):
        return self._compute_r_squared()

    @property
    def adj_r_squared(self):
        return self._compute_adj_r_squared()

    @property
    def coefficients_without_intercept(self):
        return self.model.weight.detach().cpu().numpy()

    @property
    def standard_error_of_coefficients(self):
        return self._compute_standard_error().detach().cpu().numpy()

    @property
    def y_pred(self):
        return self.model(self.x)

    @property
    def residuals(self):
        return self.y - self.y_pred

    @property
    def residuals_numpy(self):
        return self.residuals.detach().cpu().numpy()

    @property
    def y_pred_numpy(self):
        return self.y_pred.detach().cpu().numpy()

    @property
    def residual_variance(self):
        return torch.var(self.residuals)

    def _compute_standard_error(self):
        x = self.design_matrix_with_intercept
        return torch.sqrt(torch.diag(torch.inverse(x.T @ x) * self.residual_variance))

    def _compute_r_squared(self):
        ss_residual = torch.sum((self.y - self.y_pred) ** 2)
        ss_total = torch.sum((self.y - torch.mean(self.y)) ** 2)
        return 1 - (ss_residual / ss_total)

    def _compute_adj_r_squared(self):
        n = self.x.shape[0]
        p = self.x.shape[1]
        adj_r_squared = 1 - ((1 - self.r_squared) * (n - 1) / (n - p - 1))
        return adj_r_squared

    def predict(self, x_test):
        with torch.no_grad():
            x_test = torch.tensor(x_test, dtype=torch.float32)
            return self.model(x_test).numpy()

    def normality_of_residuals(self):
        _, p_value = shapiro(self.residuals.detach().cpu().numpy())
        return p_value

    def heteroscedasticity(self):
        _, p_value = bartlett(
            self.y.detach().cpu().numpy().flatten(),
            self.residuals.detach().cpu().numpy().flatten(),
        )
        return p_value

    def __str__(self):
        return self._pretty_print_coefficients() + tabulate(
            [
                ["R-squared", self.r_squared],
                ["Adjusted R-squared", self.adj_r_squared],
                ["Shapiro Wilk Statistic", self.normality_of_residuals()],
                ["Bartlett Statistic p-value", self.heteroscedasticity()],
            ],
            headers=["Metric", "Value"],
        )


# Example usage:
# x = np.random.randn(100, 2)
# y = np.random.randn(100)
# model = RegressionUtility(x, y, RegressionType.Linear)
# model.fit()
# model.display_results()
# import statsmodels.api as sm
# import polars as pl

# df = pl.read_csv("FF3.CSV")
# x = df.select(pl.col("SMB"), pl.col("HML"), pl.col("RF")).to_numpy()
# y = df.select(pl.col("Mkt-RF")).to_numpy()
# # model = sm.OLS(y, sm.add_constant(x))

# regression = RegressionUtility(
#     device="cuda",
#     x=x,
#     y=y,
#     regression_type=RegressionType.Linear,
#     epochs=50000,
#     delta_to_early_stop=1e-20,
#     learning_rate=0.01,
# )
# regression.fit()
# print(regression)
