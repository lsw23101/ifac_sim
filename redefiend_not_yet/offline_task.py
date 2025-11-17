
import numpy as np
from scipy.io import loadmat
from enc_func import *
from sympy import isprime

def compute_offline_mats(env, s=10000, num_channels=60):
    data = loadmat('FGH_data.mat')
    F_ = data['F_bar']
    G_ = data['G_']
    H_ = data['H']

    # 양자화
    
    F_bar_float = F_
    G_bar_float = np.rint(s * G_)
    H_bar_float = np.rint(s * H_)

    F_bar = np.vectorize(int)(F_bar_float)
    G_bar = np.vectorize(int)(G_bar_float)
    H_bar = np.vectorize(int)(H_bar_float)

    # 채널 수만큼 저장할 배열 예시 (axis 0 = 채널 index)
    T1_all = []
    T2_all = []
    V1_all = []
    V2_all = []
    S_xi_all = []
    S_v_all = []
    Psi_all = []
    Sigma_all = []
    Sigma_pinv_all = []

    for j in range(num_channels):
        # 예: H의 j번째 행을 쓴다든지, 지금 설계에 맞게 선택
        H1 = H_bar[j, :].copy()

        T1, T2, T, V, V1, V2 = build_TV(H1, env.q)

        S_1  = Mod(T1 @ F_bar @ V1, env.q)
        S_2  = Mod(T1 @ F_bar @ V2, env.q)
        S_3  = Mod(T1 @ G_bar,      env.q)

        Psi  = Mod(H1.reshape(1,-1) @ F_bar @ V1, env.q)
        Gamma = Mod(H1.reshape(1,-1) @ F_bar @ V2, env.q)
        Sigma = Mod(H1.reshape(1,-1) @ G_bar,      env.q)

        sigma0 = int(Sigma[0, 0])
        if sigma0 == 0:
            raise ValueError(f"[채널 {j}] Sigma[0,0] ≡ 0 (mod q), pinv 구성 불가")
        inv_sigma0 = pow(sigma0, -1, env.q)
        Sigma_pinv = np.zeros((6, 1), dtype=object)
        Sigma_pinv[0, 0] = inv_sigma0

        S_xi = Mod(S_1 - S_3 @ Sigma_pinv @ Psi, env.q)
        S_v  = Mod(S_3 @ (np.zeros((6, 6), dtype=object) - Sigma_pinv@Sigma), env.q)

        T1_all.append(T1)
        T2_all.append(T2)
        V1_all.append(V1)
        V2_all.append(V2)
        S_xi_all.append(S_xi)
        S_v_all.append(S_v)
        Psi_all.append(Psi)
        Sigma_all.append(Sigma)
        Sigma_pinv_all.append(Sigma_pinv)

    # axis 0 에 채널 index가 오도록 stack
    T1_all = np.stack(T1_all, axis=0)
    T2_all = np.stack(T2_all, axis=0)
    V1_all = np.stack(V1_all, axis=0)
    V2_all = np.stack(V2_all, axis=0)
    S_xi_all = np.stack(S_xi_all, axis=0)
    S_v_all = np.stack(S_v_all, axis=0)
    Psi_all = np.stack(Psi_all, axis=0)
    Sigma_all = np.stack(Sigma_all, axis=0)
    # Sigma_pinv_all 은 shape이 약간 다를 수 있으니 obj 배열로 둘 수도 있음

    offline = {
        "F_bar": F_bar,
        "G_bar": G_bar,
        "H_bar": H_bar,
        "T1_all": T1_all,
        "T2_all": T2_all,
        "V1_all": V1_all,
        "V2_all": V2_all,
        "S_xi_all": S_xi_all,
        "S_v_all": S_v_all,
        "Psi_all": Psi_all,
        "Sigma_all": Sigma_all,
        "Sigma_pinv_all": Sigma_pinv_all,
    }
    return offline


def save_offline_mats(offline, filename="offline_mats.npz"):
    np.savez(filename, **offline)


def load_offline_mats(filename="offline_mats.npz"):
    data = np.load(filename, allow_pickle=True)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    env = Params()
    offline = compute_offline_mats(env)
    save_offline_mats(offline)
    print("오프라인 행렬 저장 완료")
