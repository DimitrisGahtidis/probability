import numpy as np
from typing import Tuple

def kf_predict(m: np.ndarray, P: np.ndarray, A: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m_ = A @ m
        P_ = A @ P @ A.T + Q
        return m_, P_

def kf_update(m_: np.ndarray, P_: np.ndarray, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    S = H @ P_ @ H.T + R
    K = np.linalg.lstsq(S.T, H @ P_)[0].T
    m = m_ + K @ (z - H @ m_)
    P = P_ - K @ H @ P_
    return m, P

def kf_smooth(ms: np.ndarray, Ps: np.ndarray, m: np.ndarray, P: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    G = np.linalg.lstsq(P, A @ Ps)[0].T
    ms = m + G @ (ms - m)
    Ps = P + G @ (Ps - P) @ G.T
    return ms, Ps
        