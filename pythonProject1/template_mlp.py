import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import time
import copy


class MLPApproximator(BaseEstimator, RegressorMixin):
    ALGO_NAMES = ["sgd_simple", "sgd_momentum", "rmsprop", "adam"]

    def __init__(self, structure=[16, 8, 4], activation_name="relu", targets_activation_name="linear",
                 initialization_name="uniform",
                 algo_name="sgd_simple", learning_rate=1e-2, n_epochs=100, batch_size=10, seed=0,
                 verbosity_e=100, verbosity_b=10):
        self.structure = structure
        self.activation_name = activation_name
        self.targets_activation_name = targets_activation_name
        self.initialization_name = initialization_name
        self.algo_name = algo_name
        if self.algo_name not in self.ALGO_NAMES:
            self.algo_name = self.ALGO_NAMES[0]
        self.loss_name = "squared_loss"
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbosity_e = verbosity_e
        self.verbosity_b = verbosity_b
        self.history_weights = {}
        self.history_weights0 = {}
        self.n_params = None
        # params / constants for algorithms
        self.momentum_beta = 0.9
        self.rmsprop_beta = 0.9
        self.rmsprop_epsilon = 1e-7
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-7

    def __str__(self):
        txt = f"{self.__class__.__name__}(structure={self.structure},"
        txt += "\n" if len(self.structure) > 32 else " "
        txt += f"activation_name={self.activation_name}, targets_activation_name={self.targets_activation_name}, initialization_name={self.initialization_name}, "
        txt += f"algo_name={self.algo_name}, learning_rate={self.learning_rate}, n_epochs={self.n_epochs}, batch_size={self.batch_size})"
        if self.n_params:
            txt += f" [n_params: {self.n_params}]"
        return txt

    @staticmethod
    def he_uniform(n_in, n_out):
        scaler = np.sqrt(6.0 / n_in)
        return ((np.random.rand(n_out, n_in) * 2.0 - 1.0) * scaler).astype(np.float32)

    @staticmethod
    def he_normal(n_in, n_out):
        scaler = np.sqrt(2.0 / n_in)
        return (np.random.randn(n_out, n_in) * scaler).astype(np.float32)

    @staticmethod
    def glorot_uniform(n_in, n_out):
        scaler = np.sqrt(6.0 / (n_in + n_out))
        return ((np.random.rand(n_out, n_in) * 2.0 - 1.0) * scaler).astype(np.float32)

    @staticmethod
    def glorot_normal(n_in, n_out):
        scaler = np.sqrt(2.0 / (n_in + n_out))
        return (np.random.randn(n_out, n_in) * scaler).astype(np.float32)

    @staticmethod
    def prepare_batch_ranges(m, batch_size):
        n_batches = int(np.ceil(m / batch_size))
        batch_ranges = batch_size * np.ones(n_batches, dtype=np.int32)
        remainder = m % batch_size
        if remainder > 0:
            batch_ranges[-1] = remainder
        batch_ranges = np.r_[0, np.cumsum(batch_ranges)]
        return n_batches, batch_ranges

    @staticmethod
    def sigmoid(S):
        return 1 / (1 + np.exp(-S))

    @staticmethod
    def sigmoid_d(phi_S):
        return phi_S * (1 - phi_S)

    @staticmethod
    def relu(S):
        result = np.zeros_like(S)
        # ndindex - poruszanie po wszystkich indeksach tablicy
        for index in np.ndindex(S.shape):
            if S[index] > 0:
                result[index] = S[index]
            else:
                result[index] = 0

        return result

    @staticmethod
    def relu_d(phi_S):
        result = np.zeros_like(phi_S, dtype=np.float32)
        #ndindex - poruszanie po wszystkich indeksach tablicy
        for index in np.ndindex(phi_S.shape):
            if phi_S[index] > 0:
                result[index] = 1.0
            else:
                result[index] = 0.0
        return result


    @staticmethod
    def linear(S):
        return S

    @staticmethod
    def linear_d(phi_S):
        return np.ones_like(phi_S)

    @staticmethod
    def squared_loss(y_MLP, y_target):
        losses = np.zeros(y_MLP.shape[0])
        for i in range(y_MLP.shape[0]):
            squared_sum = 0.0
            for j in range(y_MLP.shape[1]):
                diff = y_MLP[i][j] - y_target[i][j]
                squared_sum += diff ** 2
            losses[i] = 0.5 * squared_sum
        return losses

    @staticmethod
    def squared_loss_d(y_MLP, y_target):
        return y_MLP - y_target

    def pre_algo_sgd_simple(self):
        return  # no special preparation needed for simple SGD

    def algo_sgd_simple(self, l):
        #Zmiana wag
        # waga(l) = waga(l) - learning_rate * gradient(l)
        delta = self.learning_rate * self.gradients[l]
        # Aktualizacja wag dla warstwy l
        #waga(l) = waga(l) - delta
        self.weights_[l] -= delta

        # Aktualizacja wag dla biasow warstwy l:
        #waga0(l) = waga0(l) - learning_rate * gradient0(l)
        self.weights0_[l] -= self.learning_rate * self.gradients0[l]

    def pre_algo_sgd_momentum(self):
        self.velocity = []  #momentum dla wag
        self.velocity0 = []  #momentum dla biasów
        for w in self.weights_:
            # inicjalizacja momentum wag na 0
            self.velocity.append(np.zeros_like(w))

        for w0 in self.weights0_:
            # inicjalizacja momentum biasów na 0
            self.velocity0.append( np.zeros_like(w0))

    def algo_sgd_momentum(self, l):
        #Momentum dla wag: v_l = beta * v_l - lr * gradient_w
        self.velocity[l] = self.momentum_beta * self.velocity[l] - self.learning_rate * self.gradients[l]
        #Aktualizacja wag: w_l = w_l + v_l
        self.weights_[l] += self.velocity[l]
        #Momentum dla biasów: v_0 = beta * v_0 - lr * gradient_b
        self.velocity0[l] = self.momentum_beta * self.velocity0[l] - self.learning_rate * self.gradients0[l]
        #Aktualizacja biasow: b_0 = b_0 + v_0
        self.weights0_[l] += self.velocity0[l]

    def pre_algo_rmsprop(self):
        self.s = []
        self.s0 = []
        for w in self.weights_:
            # Inicjalizacja sumy kwadratów gradientów dla wag na 0
            self.s.append(np.zeros_like(w))
        for w0 in self.weights0_:
            # Inicjalizacja sumy kwadratów gradientów dla biasów na 0
            self.s0.append(np.zeros_like(w0))

    def algo_rmsprop(self, l):
        #suma kwadratów gradientów dla wag: s_l = beta * s_l + (1 - beta) * (gradient_w)^2
        self.s[l] = self.rmsprop_beta * self.s[l] + (1 - self.rmsprop_beta) * self.gradients[l] ** 2

        #suma kwadratów gradientów dla biasów: s_0 = beta * s_0 + (1 - beta) * (gradient_b)^2
        self.s0[l] = self.rmsprop_beta * self.s0[l] + (1 - self.rmsprop_beta) * self.gradients0[l] ** 2

        #Nowa wartość wag
        self.weights_[l] -= (self.learning_rate / (np.sqrt(self.s[l]) + self.rmsprop_epsilon)) * self.gradients[l]

        # Aktualizacja biasów: b_0 = b_0 - (learning_rate / (sqrt(s_0) + epsilon)) * gradient_b
        self.weights0_[l] -= (self.learning_rate / (np.sqrt(self.s0[l]) + self.rmsprop_epsilon)) * self.gradients0[l]

    def pre_algo_adam(self):
        self.m = []
        self.v = []
        self.m0 = []
        self.v0 = []
        for w in self.weights_:
            # Inicjalizacja momentu wag na 0
            self.m.append(np.zeros_like(w))
            # Inicjalizacja drugiego momentu wag na 0
            self.v.append(np.zeros_like(w))

        for w0 in self.weights0_:
            # Inicjalizacja momentu biasów na 0
            self.m0.append(np.zeros_like(w0))
            # Inicjalizacja drugiego momentu biasów na 0
            self.v0.append(np.zeros_like(w0))

        self.t = 0  #licznik iteracji

    def algo_adam(self, l):
        self.t += 1 #inkrementacja licznika iteracji

        grad = self.gradients[l]  # Gradient wag
        grad0 = self.gradients0[l]  # Gradient biasów

        # Aktualizacja pierwszego momentu (momentum) dla wag
        # m_l = beta1 * m_l + (1 - beta1) * grad_w
        self.m[l] = self.adam_beta1 * self.m[l] + (1 - self.adam_beta1) * grad

        # Aktualizacja drugiego momentu (kwadratu gradientów) dla wag
        # v_l = beta2 * v_l + (1 - beta2) * grad_w^2
        self.v[l] = self.adam_beta2 * self.v[l] + (1 - self.adam_beta2) * grad ** 2

        # Korekcja biasów dla pierwszego momentu (m_l)
        m_hat = self.m[l] / (1 - self.adam_beta1 ** self.t)

        # Korekcja biasów dla drugiego momentu (v_l)
        v_hat = self.v[l] / (1 - self.adam_beta2 ** self.t)

        # Aktualizacja wag
        # w_l = w_l - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        self.weights_[l] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.adam_epsilon)

        # Aktualizacja pierwszego momentu (momentum) dla biasów
        # m0_l = beta1 * m0_l + (1 - beta1) * grad_b
        self.m0[l] = self.adam_beta1 * self.m0[l] + (1 - self.adam_beta1) * grad0

        # Aktualizacja drugiego momentu (kwadratu gradientów) dla biasów
        # v0_l = beta2 * v0_l + (1 - beta2) * grad_b^2
        self.v0[l] = self.adam_beta2 * self.v0[l] + (1 - self.adam_beta2) * grad0 ** 2

        # Korekcja biasów dla pierwszego momentu (m0_l)
        # m_hat0 = m0_l / (1 - beta1^t)
        m_hat0 = self.m0[l] / (1 - self.adam_beta1 ** self.t)

        # Korekcja biasów dla drugiego momentu (v0_l)
        # v_hat0 = v0_l / (1 - beta2^t)
        v_hat0 = self.v0[l] / (1 - self.adam_beta2 ** self.t)

        # Aktualizacja biasów
        # b0_l = b0_l - learning_rate * m_hat0 / (sqrt(v_hat0) + epsilon)
        self.weights0_[l] -= self.learning_rate * m_hat0 / (np.sqrt(v_hat0) + self.adam_epsilon)

    def fit(self, X, y):
        np.random.seed(self.seed)
        self.activation_ = getattr(MLPApproximator, self.activation_name)
        self.activation_d_ = getattr(MLPApproximator, self.activation_name + "_d")
        self.initialization_ = getattr(MLPApproximator, (
            "he_" if self.activation_name == "relu" else "glorot_") + self.initialization_name)
        self.targets_activation_ = getattr(MLPApproximator, self.targets_activation_name)
        self.targets_activation_d_ = getattr(MLPApproximator, self.targets_activation_name + "_d")
        self.loss_ = getattr(MLPApproximator, self.loss_name)
        self.loss_d_ = getattr(MLPApproximator, self.loss_name + "_d")
        self.pre_algo_ = getattr(self, "pre_algo_" + self.algo_name)
        self.algo_ = getattr(self, "algo_" + self.algo_name)
        self.weights_ = [
            None]
        self.weights0_ = [
            None]
        m, n = X.shape
        if len(y.shape) == 1:
            y = np.array([y]).T
        self.n_ = n
        self.n_targets_ = 1 if len(y.shape) == 1 else y.shape[1]
        self.n_params = 0
        for l in range(len(self.structure) + 1):
            n_in = n if l == 0 else self.structure[l - 1]
            n_out = self.structure[l] if l < len(self.structure) else self.n_targets_
            w = self.initialization_(n_in, n_out)
            w0 = np.zeros((n_out, 1), dtype=np.float32)
            self.weights_.append(w)
            self.weights0_.append(w0)
            self.n_params += w.size
            self.n_params += w0.size
        t1 = time.time()
        if self.verbosity_e > 0:
            print(f"FIT [total of weights (params): {self.n_params}]")
        self.pre_algo_()  # if some preparation needed
        n_batches, batch_ranges = MLPApproximator.prepare_batch_ranges(m, self.batch_size)
        self.t = 0
        for e in range(self.n_epochs):
            t1_e = time.time()
            if e % self.verbosity_e == 0 or e == self.n_epochs - 1:
                print("-" * 3)
                print(f"EPOCH {e + 1}/{self.n_epochs}:")
                self.forward(X)
                loss_e_before = np.mean(self.loss_(self.signals[-1], y))
            p = np.random.permutation(m)
            for b in range(n_batches):
                indexes = p[batch_ranges[b]: batch_ranges[b + 1]]
                X_b = X[indexes]
                y_b = y[indexes]
                self.forward(X_b)
                loss_b_before = np.mean(self.loss_(self.signals[-1], y_b))
                self.backward(y_b)
                for l in range(1, len(self.structure) + 2):
                    self.algo_(l)
                if (e % self.verbosity_e == 0 or e == self.n_epochs - 1) and b % self.verbosity_b == 0:
                    self.forward(X_b)
                    loss_b_after = np.mean(self.loss_(self.signals[-1], y_b))
                    print(
                        f"[epoch {e + 1}/{self.n_epochs}, batch {b + 1}/{n_batches} -> loss before: {loss_b_before}, loss after: {loss_b_after}]")
                self.t += 1
            t2_e = time.time()
            if e % self.verbosity_e == 0 or e == self.n_epochs - 1:
                self.forward(X)
                loss_e_after = np.mean(self.loss_(self.signals[-1], y))
                self.history_weights[e] = copy.deepcopy(self.weights_)
                self.history_weights0[e] = copy.deepcopy(self.weights0_)
                print(
                    f"ENDING EPOCH {e + 1}/{self.n_epochs} [loss before: {loss_e_before}, loss after: {loss_e_after}; epoch time: {t2_e - t1_e} s]")
        t2 = time.time()
        if self.verbosity_e > 0:
            print(f"FIT DONE. [time: {t2 - t1} s]")

    def forward(self, X_b):
        self.signals = [None] * (len(self.structure) + 2)
        self.signals[0] = X_b
        for l in range(1, len(self.signals)):
            S = np.dot(self.signals[l - 1], self.weights_[l].T) + self.weights0_[l].T
            phi_S = self.activation_(S) if l < len(self.signals) - 1 else self.targets_activation_(S)
            self.signals[l] = phi_S

    def backward(self, y_b):
        # Inicjalizacja list
        self.deltas = [None] * len(self.signals)
        self.gradients = [None] * len(self.signals)
        self.gradients0 = [None] * len(self.signals)

        #indeks warstwy wyjsciowej
        l = len(self.signals) - 1

        # Obliczanie delty na wyjściu
        self.deltas[l] = self.loss_d_(self.signals[l], y_b) * self.targets_activation_d_(self.signals[l])

        # Obliczanie gradientu wag dla wyjscia
        self.gradients[l] = np.dot(self.deltas[l].T, self.signals[l - 1])

        # Obliczanie gradientu dla biasów warstwy wyjściowej
        self.gradients0[l] = np.sum(self.deltas[l], axis=0, keepdims=True).T

        # Propagacja błędu wstecz przez warstwy ukryte
        for l in range(len(self.signals) - 2, 0, -1):  #iterujemy w tył po warstwach
            # Delta warstwy l
            delta = np.dot(self.deltas[l + 1], self.weights_[l + 1]) * self.activation_d_(self.signals[l])
            self.deltas[l] = delta

            #Gradient wag dla warstwy l
            self.gradients[l] = np.dot(delta.T, self.signals[l - 1])

            #Gradient dla biasów warstwy l
            self.gradients0[l] = np.sum(delta, axis=0, keepdims=True).T

    def predict(self, X):
        self.forward(X)
        y_pred = self.signals[-1]
        if self.n_targets_ == 1:
            y_pred = y_pred[:, 0]
        return y_pred