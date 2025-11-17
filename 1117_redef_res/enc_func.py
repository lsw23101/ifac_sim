import numpy as np
import random


# 이 파일이 해야하는 일:
# 비밀키 만들기
# 모듈러, 암호화 (초기값, 출력), 복호화 (24길이 혹은 길이 1 짜리 둘 다 호환 되도록)


class Params:
    def __init__(self):
        # 여기서 K의 최댓값은 355,700,000 (참고용)
        self.p = int(2**44)   # p 
        self.L = int(2**20)   # L 
        self.r = 19           # 오류 범위 # 균등분포
        self.N = 5            # 키 차원 # 연산량...
        # 2^64 근처 소수
        self.q = self.p * self.L - 59    # 18446744073709551557
        # L의 모듈러 역원 (L^{-1} mod q) 미리 계산해서 저장
        L_mod = self.L % self.q
        self.invL = pow(L_mod, -1, self.q)   # int 타입


env = Params()

def Seret_key(env):
    # -1 0 1 균등 분포
    sk = np.array([random.choice([-1, 0, 1]) for _ in range(env.N)], dtype=object)
    # print(sk)
    return sk.reshape(-1, 1)  # N x 1 형태로 반환

def Mod(x, p):
    # -q/2 q/2 인 mod 연산
    x_arr = np.asarray(x, dtype=object)

    def centered(v):
        v_int = int(v)
        r = v_int % p
        if r >= p // 2:
            r -= p
        return r

    return np.vectorize(centered, otypes=[object])(x_arr)

def rand_mod_q(shape, q):
    """
    shape 모양의 배열에 대해 [0, q-1] 에서 균일분포 랜덤 정수 생성 (dtype=object)
    np.random.randint 로는 q가 너무 커서 안 되므로 Python random 사용
    """
    arr = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        arr[idx] = random.randrange(q)
    return arr



def Enc_state(z_hat_bar, sk, env, T1, T2, V2):
    """
    Enc_state, j 인덱스에 따라 60개

    입력:
        z_hat_bar : (24x1) 양자화된 초기 옵저버 상태
        sk        : (N x 1) 비밀키
        env       : Params()
        T1        : 23x24 
        T2        : 1x24  
        V2        : 24x1  

    출력:
        C_state   : 24 x (N+2) 암호문
        b_xi_ini  : 23 x 1 Enc_t 를 위한 동적 마스킹 파트
    """
    n = 24
    N = env.N

    z_hat_bar = np.asarray(z_hat_bar, dtype=object).reshape(n, 1)
    sk = np.asarray(sk, dtype=object).reshape(N, 1)

    # 1) A, e 
    A = rand_mod_q((n, N), env.q)   # 24 x N, entries in [0, q-1]
    # A = np.random.randint(0, 50, size=(n, N)).astype(object)

    # e = np.random.randint(-env.r, env.r + 1, size=(n, 1)).astype(object)  # 6x1 에러주입
    e = np.zeros((n, 1), dtype=object) # 테스트용 에러 0

    # 2) b_ini = A sk + e
    b_ini = Mod(A @ sk + e, env.q)

    # 3) b_tilde, b_xi_ini
    b_tilde = T2 @ b_ini           # 1x1
    b_tilde = Mod(b_tilde, env.q)

    b_xi_ini = T1 @ b_ini          # 23x1
    b_xi_ini = Mod(b_xi_ini, env.q)

    # 4) b_prime = V2 @ b_tilde
    b_prime = V2 @ b_tilde         # 24x1
    b_prime = Mod(b_prime, env.q)

    # 5) 첫 번째 컬럼
    C0 = Mod(env.L * z_hat_bar + b_ini - b_prime, env.q)  # 24x1

    # 6) 최종 암호문
    C_state = np.hstack([C0, A, b_prime])         # 24 x (N+2)

    return C_state, b_xi_ini


def Enc_t(v, sk, b_xi, Sigma_pinv, Sigma, Psi, env):
    """
    Enc_t (동적 암호화)

    입력:
        v           : 6x1 벡터 (양자화된 [u;y])
        sk          : N x 1 비밀키
        b_xi        : 23 x 1 (마스킹 상태, 업데이트는 함수 밖에서)
        S_xi, S_v   : 23x23, 23x6 (지금은 함수 안에서 사용 X, 추후 b_xi 업데이트용)
        Sigma_pinv  : 6 x 1
        Sigma       : 1 x 6
        Psi         : 1 x 23
        env         : Params()

    출력:
        C_t         : 6 x (N+2) 암호문 [v + b_v - b_prime | Av | b_prime]
    """
    n_v = 6
    N = env.N

    # 형 맞추기
    v = np.asarray(v, dtype=object).reshape(n_v, 1)
    sk = np.asarray(sk, dtype=object).reshape(N, 1)
    b_xi = np.asarray(b_xi, dtype=object).reshape(23, 1)
    Sigma = np.asarray(Sigma, dtype=object).reshape(1, 6)
    Sigma_pinv = np.asarray(Sigma_pinv, dtype=object).reshape(6, 1)
    Psi = np.asarray(Psi, dtype=object).reshape(1, 23)

    # 1) Av, e 생성
    # Av = rand_mod_q((n_v, N), env.q)   # 6 x N, entries in [0, q-1]
    Av= np.random.randint(0, 1000000, size=(n_v, N)).astype(object)
    # e = np.random.randint(-env.r, env.r + 1, size=(n_v, 1)).astype(object)  # 6x1              
    e = np.zeros((n_v, 1), dtype=object)

    # 2) b_v = Av sk + e (6x1)
    b_v = Mod(Av @ sk + e, env.q)

    # 3) b_prime = Sigma_pinv @ (Sigma @ b_v + Psi @ b_xi) (6x1)
    tmp = Mod(Sigma @ b_v + Psi @ b_xi , env.q)
    b_prime = Sigma_pinv @ tmp       # 6x1
    b_prime = Mod(b_prime, env.q)

    # 4) 첫 번째 컬럼: c0 = v + b_v - b_prime (6x1, mod q)
    c0 = Mod(env.L * v + b_v - b_prime, env.q)

    # 5) 최종 암호문: [c0 | Av | b_prime] (6 x (N+2))
    C_t = np.hstack([c0, Av, b_prime])

    # 디버깅용 출력 (원하면 주석처리)
    # print("[Enc_t] C_t shape:", C_t.shape)

    return C_t, b_v


def Dec(ciphertext, sk, env):
    
    dec_sk = np.vstack([
        np.ones((1, 1), dtype=object),
        -sk,
        np.ones((1, 1), dtype=object)
    ])
    
    # 1) 모듈러 곱 (0..q-1)
    m_bar = Mod(ciphertext@ dec_sk, env.q)  # (h x 1)

    # 2) centered 로 다시 잡기 [-q/2, q/2)
    m_center = Mod(m_bar, env.q)   # 여기서 Mod는 네가 정의한 centered 함수
    
    # 3) 스케일링은 호출하는 쪽에서 알아서 (L, r_quant, s_quant 등)
    return m_center


