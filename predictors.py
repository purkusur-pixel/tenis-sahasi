"""
predictors.py
-------------
Oyuncu konum TAHMIN modelleri (secilebilir). Hepsi kort duzleminde (metre)
calisir ve ortak arayuzu paylasir:

    reset(meas)      -> ilk olcumle durumu baslat
    predict()        -> bir sonraki kareyi kestir (a-priori), (x, y) doner
    update(meas)     -> olcumle duzelt, (x, y) doner
    predicted()      -> olcum yoksa kullanilacak a-priori kestirim (x, y)

dt = config.FRAME_SKIP (kare birimi). Olcum gurultusu ANIZOTROPIK'tir
(config.KF_MEASUREMENT_NOISE = x, KF_MEASUREMENT_NOISE_Y = y): uzak yari
derinlik cozunurlugu dusuk oldugundan y olcumune daha az guvenilir.

Modeller (menuden secilir, config.PREDICT_TYPE):
    CV  - Sabit Hiz (Kalman, [x,y,vx,vy])            : baseline, en ucuz
    CA  - Sabit Ivme (Kalman, [x,y,vx,vy,ax,ay])     : hizlanma/yavaslama
    ABG - alfa-beta-gama sabit-kazancli filtre       : hafif, ayari kolay
    IMM - Etkilesimli Coklu Model (2 mod: dengeli+manevra) : en saglam
"""

import numpy as np
import cv2

import config


def _R():
    """Anizotropik olcum gurultu kovaryansi (x, y)."""
    return np.array([[config.KF_MEASUREMENT_NOISE, 0.0],
                     [0.0, config.KF_MEASUREMENT_NOISE_Y]], dtype=np.float32)


# =============================================================================
# cv2.KalmanFilter sarmalayici (CV ve CA ortak akis)
# =============================================================================
class _CvKalman:
    def __init__(self, n):
        self.n = n
        self.kf = cv2.KalmanFilter(n, 2, 0, cv2.CV_32F)
        self._pred = (0.0, 0.0)

    def reset(self, meas):
        s = np.zeros((self.n, 1), dtype=np.float32)
        s[0, 0], s[1, 0] = float(meas[0]), float(meas[1])
        self.kf.statePost = s
        self.kf.errorCovPost = np.eye(self.n, dtype=np.float32) * config.KF_ERROR_COV
        self._pred = (float(meas[0]), float(meas[1]))

    def predict(self):
        p = self.kf.predict()
        self._pred = (float(p[0, 0]), float(p[1, 0]))
        return self._pred

    def update(self, meas):
        e = self.kf.correct(np.array([[float(meas[0])], [float(meas[1])]], dtype=np.float32))
        return (float(e[0, 0]), float(e[1, 0]))

    def predicted(self):
        return self._pred


class CV(_CvKalman):
    """Sabit Hiz: durum [x, y, vx, vy]."""
    def __init__(self):
        super().__init__(4)
        dt = float(config.FRAME_SKIP)
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                             [0, 1, 0, dt],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * config.KF_PROCESS_NOISE
        self.kf.measurementNoiseCov = _R()


class CA(_CvKalman):
    """Sabit Ivme: durum [x, y, vx, vy, ax, ay]."""
    def __init__(self):
        super().__init__(6)
        dt = float(config.FRAME_SKIP)
        h = 0.5 * dt * dt
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0, h, 0],
                                             [0, 1, 0, dt, 0, h],
                                             [0, 0, 1, 0, dt, 0],
                                             [0, 0, 0, 1, 0, dt],
                                             [0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * config.KF_PROCESS_NOISE
        self.kf.measurementNoiseCov = _R()


# =============================================================================
# alfa-beta-gama sabit-kazancli filtre (Kalman'in hafif hali)
# =============================================================================
class ABG:
    """alfa-beta-gama: her eksende [konum, hiz, ivme], sabit kazanclar."""
    def __init__(self):
        self.a = float(config.ABG_ALPHA)
        self.b = float(config.ABG_BETA)
        self.g = float(config.ABG_GAMMA)
        self.dt = float(config.FRAME_SKIP)
        self.p = [0.0, 0.0]
        self.v = [0.0, 0.0]
        self.ac = [0.0, 0.0]
        self._pred = (0.0, 0.0)

    def reset(self, meas):
        self.p = [float(meas[0]), float(meas[1])]
        self.v = [0.0, 0.0]
        self.ac = [0.0, 0.0]
        self._pred = (self.p[0], self.p[1])

    def predict(self):
        dt = self.dt
        for i in (0, 1):
            self.p[i] += self.v[i] * dt + 0.5 * self.ac[i] * dt * dt
            self.v[i] += self.ac[i] * dt
        self._pred = (self.p[0], self.p[1])
        return self._pred

    def update(self, meas):
        dt = self.dt
        for i in (0, 1):
            r = float(meas[i]) - self.p[i]          # kalinti (residual)
            self.p[i] += self.a * r
            self.v[i] += (self.b / dt) * r
            self.ac[i] += (2.0 * self.g / (dt * dt)) * r
        return (self.p[0], self.p[1])

    def predicted(self):
        return self._pred


# =============================================================================
# IMM - Etkilesimli Coklu Model (2 mod: DENGELI + MANEVRA)
# =============================================================================
class _LKF:
    """IMM alt-modeli icin elle (numpy) lineer Kalman; likelihood da uretir."""
    def __init__(self, F, H, Q, R):
        self.F, self.H, self.Q, self.R = F, H, Q, R
        self.x = None
        self.P = None
        self.L = 1.0

    def set(self, x, P):
        self.x = x.copy()
        self.P = P.copy()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x                       # yenilik (innovation)
        S = self.H @ self.P @ self.H.T + self.R
        Sinv = np.linalg.inv(S)
        K = self.P @ self.H.T @ Sinv
        self.x = self.x + K @ y
        self.P = (np.eye(self.x.shape[0]) - K @ self.H) @ self.P
        d = float(y.T @ Sinv @ y)
        det = float(np.linalg.det(2.0 * np.pi * S))
        self.L = np.exp(-0.5 * d) / np.sqrt(max(det, 1e-12))


class IMM:
    """
    Iki mod: (1) DENGELI (dusuk process noise -> duz/sabit hiz), (2) MANEVRA
    (yuksek process noise -> ani hiz/yon degisimi). Her karede hangi mod uyuyorsa
    ona agirlik verilir. Tenis'te oyuncu durur/sprint atar/yon degistirir -> en saglam.
    """
    def __init__(self):
        dt = float(config.FRAME_SKIP)
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float64)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float64)
        R = np.array([[config.KF_MEASUREMENT_NOISE, 0.0],
                      [0.0, config.KF_MEASUREMENT_NOISE_Y]], dtype=np.float64)
        q = float(config.KF_PROCESS_NOISE)
        qm = q * float(config.IMM_MANEUVER_FACTOR)
        self.models = [_LKF(F, H, np.eye(4) * q, R),      # dengeli
                       _LKF(F, H, np.eye(4) * qm, R)]     # manevra
        self.mu = np.array([0.5, 0.5])
        p = float(config.IMM_STAY)                        # ayni modda kalma olasiligi
        self.PI = np.array([[p, 1.0 - p], [1.0 - p, p]])
        self._cbar = self.mu.copy()
        self._pred = (0.0, 0.0)

    def reset(self, meas):
        x = np.array([float(meas[0]), float(meas[1]), 0.0, 0.0])
        P = np.eye(4) * float(config.KF_ERROR_COV)
        for m in self.models:
            m.set(x, P)
        self.mu = np.array([0.5, 0.5])
        self._pred = (float(meas[0]), float(meas[1]))

    def predict(self):
        n = len(self.models)
        cbar = self.PI.T @ self.mu                       # cbar[j] = sum_i PI[i,j] mu[i]
        mix = (self.PI * self.mu[:, None]) / np.maximum(cbar[None, :], 1e-12)
        xs = [m.x for m in self.models]
        Ps = [m.P for m in self.models]
        for j, m in enumerate(self.models):
            x0 = sum(mix[i, j] * xs[i] for i in range(n))
            P0 = sum(mix[i, j] * (Ps[i] + np.outer(xs[i] - x0, xs[i] - x0)) for i in range(n))
            m.set(x0, P0)
            m.predict()
        self._cbar = cbar
        x = sum(self.mu[j] * self.models[j].x for j in range(n))
        self._pred = (float(x[0]), float(x[1]))
        return self._pred

    def update(self, meas):
        z = np.array([float(meas[0]), float(meas[1])])
        for m in self.models:
            m.update(z)
        L = np.array([m.L for m in self.models])
        self.mu = L * self._cbar
        self.mu = self.mu / max(self.mu.sum(), 1e-12)
        x = sum(self.mu[j] * self.models[j].x for j in range(len(self.models)))
        return (float(x[0]), float(x[1]))

    def predicted(self):
        return self._pred


# =============================================================================
# FABRIKA
# =============================================================================
_REGISTRY = {"CV": CV, "CA": CA, "ABG": ABG, "IMM": IMM}


def create(kind):
    """config.PREDICT_TYPE'a gore tahminci uretir. Bilinmeyen -> CV."""
    return _REGISTRY.get(str(kind).upper(), CV)()
