

import numpy as np
from scipy.io import loadmat
from enc_func import *
from sympy import isprime


# s = 10000으로 고정해둔 상태.

# 상대차수가 모두 1 => H 행벡터 가져오고 나머지 T1은 표준기저로 설정
# 역행렬 V도 표준기저와 20번째 행이 Zq의 역원으로 계산
def build_TV(H1, q):
    """
    입력:  
        H1 : 1×24 numpy object vector  
        q  : modulus

    출력:  
        T1, T2, T, V, V1, V2

    설계:
      - pivot column = 20번째 열 (0-based index 19)
      - T1 : 23x24, 20열만 제외한 나머지 23개 열에 대한 선택행렬
      - T2 = H1
      - T  = [T1; T2]
      - V  = T^{-1} (mod q) 를 closed-form으로 직접 구성
      - V  = [V1 V2] with V1: 24x23, V2: 24x1
    """

    H1 = np.asarray(H1, dtype=object).reshape(24,)
    n = 24
    pivot = 19  # 0-based index, = 20번째 열

    # -------------------------
    # 1) T1 만들기 : "pivot 열만 제외한 표준 기저"
    # -------------------------
    T1 = np.zeros((n - 1, n), dtype=object)  # 23x24
    cols = list(range(n))
    cols.remove(pivot)   # [0,1,...,18,20,21,22,23]
    for row, col in enumerate(cols):
        T1[row, col] = 1

    # 2) T2 = H1
    T2 = H1.reshape(1, n)

    # 3) T = [T1; T2]
    T = np.vstack([T1, T2])

    # print("[build_TV] T1 shape:", T1.shape)
    # print("[build_TV] T  shape:", T.shape)

    # -------------------------
    # 4) V = T^{-1} (mod q) 
    # -------------------------
    V = np.zeros((n, n), dtype=object)

    # print("T", T)
    # print("V", V)

    # pivot 값 및 역원
    h_p = int(H1[pivot]) % q
    if h_p == 0:
        raise ValueError("H1[19] (20번째 원소) ≡ 0 (mod q) 입니다. pivot으로 사용할 수 없습니다.")
    inv_h_p = pow(h_p, -1, q)


    # 앞의 n-1개 열 (1~23열) 채우기
    #
    # 열 j (0 <= j < n-1) 는 "원래 인덱스 orig_idx"에 대응:
    #   j < pivot  ->  orig_idx = j
    #   j >= pivot ->  orig_idx = j + 1   (pivot 열을 건너뛴 것)
    #
    # 행 i != pivot:
    #   orig_idx == i 이면 V[i,j] = 1  (항등)
    #   아니면 0
    #
    # 행 i == pivot:
    #   V[pivot, j] = - H1[orig_idx] / H1[pivot]  (mod q)
    #
    for j in range(n - 1):  # 0..22
        orig_idx = j if j < pivot else j + 1  # 0-based
        h_orig = int(H1[orig_idx]) % q

        for i in range(n):
            if i == pivot:
                # pivot 행: -h_orig / h_p (mod q)
                V[i, j] = (-h_orig * inv_h_p) % q
            else:
                # identity mapping
                if i == orig_idx:
                    V[i, j] = 1
                else:
                    V[i, j] = 0

    # 마지막 열 (23, 즉 24번째 열):
    #   모든 행 0, 단 pivot 행에만 1/h_p
    for i in range(n):
        V[i, n - 1] = 0
    V[pivot, n - 1] = inv_h_p % q

    # print("inverse test:", Mod(T@V, env.q))
    # -------------------------
    # 5) V1, V2 분리
    # -------------------------
    V1 = V[:, :n - 1].copy()          # 24x23
    V2 = V[:, n - 1].reshape(n, 1)    # 24x1

    return T1, T2, T, V, V1, V2





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
        # j 인덱스마다, H의 행벡터에 대하여 좌표변환 행렬 T, V 찾기
        H1 = H_bar[j, :].copy()

        T1, T2, T, V, V1, V2 = build_TV(H1, env.q)

        # 식(19) 식
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




# npz 파일 읽고 쓰기
def save_offline_mats(offline, filename="offline_mats.npz"):
    np.savez(filename, **offline)


def load_offline_mats(filename="offline_mats.npz"):
    data = np.load(filename, allow_pickle=True)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    env = Params()
    offline = compute_offline_mats(env)
    save_offline_mats(offline)
    print("오프라인 행렬 저장 완료\n")

    ##  디버그용 프린트

    F_bar = offline["F_bar"]
    G_bar = offline["G_bar"]
    H_bar = offline["H_bar"]

    print("===== F_bar =====")
    print(F_bar)
    print("\n===== G_bar =====")
    print(G_bar)
    print("\n===== H_bar =====")
    print(H_bar)

    # === 채널 0의 오프라인 행렬들 출력 ===
    ch = 0
    print(f"\n===== Channel {ch} offline matrices =====")

    T1_all         = offline["T1_all"]
    T2_all         = offline["T2_all"]
    V1_all         = offline["V1_all"]
    V2_all         = offline["V2_all"]
    S_xi_all       = offline["S_xi_all"]
    S_v_all        = offline["S_v_all"]
    Psi_all        = offline["Psi_all"]
    Sigma_all      = offline["Sigma_all"]
    Sigma_pinv_all = offline["Sigma_pinv_all"]  # 리스트 형태

    print("\nT1:")
    print(T1_all[ch])

    print("\nT2:")
    print(T2_all[ch])

    print("\nV1:")
    print(V1_all[ch])

    print("\nV2:")
    print(V2_all[ch])

    print("\nS_xi:")
    print(S_xi_all[ch])

    print("\nS_v]:")
    print(S_v_all[ch])

    print("\nPsi:")
    print(Psi_all[ch])

    print("\nSigma:")
    print(Sigma_all[ch])

    print("\nSigma_pinv:")
    print(Sigma_pinv_all[ch])