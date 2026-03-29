"""
SparseTap — Plan B: Random Subset Gaussian Elimination (ISD Approach)
- LPN(Learning Parity with Noise) 완전 해법 (조합론 우회)
- 무작위 방정식 64개를 뽑아 가우스 소거법으로 연립방정식을 풂
- S의 개수가 10개가 넘어가도 속도가 동일한 궁극의 수학적 알고리즘
"""

import numpy as np
import torch
import time
import os

DATA_PATH = "DAY2_data.txt"
TEST_PREFIX = "0000010100011010010101100101001110100011110010110011010000111010"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    return np.array([[int(c) for c in line] for line in lines], dtype=np.uint8)

def precompute_data():
    data = load_data(DATA_PATH)
    N, L = data.shape
    W = 64
    
    y_target = data[:, W:].flatten()  # (384000,)
    X_bits = np.zeros((N * (L - W), W), dtype=np.uint8)
    
    idx = 0
    for i in range(N):
        for n in range(W, L):
            # x_{n-1} ~ x_{n-64} (문제 정의 포맷: 최신 비트가 첫번째 인덱스)
            X_bits[idx] = data[i, n-64:n][::-1] 
            idx += 1
            
    return torch.tensor(X_bits, dtype=torch.uint8, device=device), \
           torch.tensor(y_target, dtype=torch.uint8, device=device).unsqueeze(1)

def generate_answer(offsets, prefix):
    seq = [int(c) for c in prefix]
    for n in range(64, 256):
        bit = 0
        for d in offsets:
            bit ^= seq[n - d]
        seq.append(bit)
    return "".join(str(b) for b in seq[64:])

@torch.no_grad()
def solve_lpn_random_subset():
    print(f"Device: {device}")
    t0 = time.time()
    
    X_all, y_all = precompute_data()
    TOTAL_EQS = X_all.shape[0]
    W = 64
    
    print(f"데이터 로드 완료. 방정식 개수: {TOTAL_EQS:,} 개")
    print(f"20% 노이즈 환경에서 64개의 순수 방정식을 정확히 뽑을 확률은 약 6.2e-7 입니다.")
    print(f"따라서 약 5백만 번의 역행렬을 풀면 99% 확정적으로 정답을 찾습니다.\n")
    
    BATCH_SIZE = 150_000  
    MAX_ITER = 1_000       # 15만 x 1000 = 1억 5천만 번 역행렬 계산
    
    # 연산 최적화를 위해 5,000개만 먼저 체크
    TEST_SZ = 5000
    idx_test = torch.randperm(TOTAL_EQS, device=device)[:TEST_SZ]
    X_sub = X_all[idx_test].float()
    y_sub = y_all[idx_test].float().squeeze(1)

    X_all_f = X_all.float()
    y_all_f = y_all.float().squeeze(1)
    
    idx0 = torch.arange(BATCH_SIZE, device=device)
    
    for it in range(1, MAX_ITER + 1):
        it_start = time.time()
        
        # 1. 랜덤으로 BATCH_SIZE * 64 개의 방정식 추출
        picks = torch.randint(0, TOTAL_EQS, (BATCH_SIZE, W), device=device)
        A_b = X_all[picks] # (B, 64, 64)
        y_b = y_all[picks] # (B, 64, 1)
        
        # 2. 첨가 행렬 생성: (B, 64, 65)
        M = torch.cat([A_b, y_b], dim=2)
        
        # 3. GPU Batched Gaussian Elimination over GF(2)
        for i in range(W):
            col = M[:, i:, i].to(torch.int32)
            pivot_idx = col.argmax(dim=1)
            idx1 = i + pivot_idx
            
            row_i = M[idx0, i, :].clone()
            row_idx1 = M[idx0, idx1, :].clone()
            
            M[idx0, i, :] = row_idx1
            M[idx0, idx1, :] = row_i
            
            pivot_row = M[:, i:i+1, :]
            mask = M[:, :, i:i+1].clone()
            mask[idx0, i, 0] = 0
            
            # XOR row operations
            torch.bitwise_xor(M, pivot_row * mask, out=M)
            
        # 4. 역행렬(가우스 소거) 검증
        diag = M[:, range(W), range(W)]
        valid = (diag.sum(dim=1) == W)  # 대각선이 모두 1이면 풀 랭크(해 존재)
        
        S_cands = M[valid, :, 64].float() # 풀 랭크를 가진 정답 벡터들 후보 (V, 64)
        V = S_cands.shape[0]
        
        if V == 0:
            continue
            
        # 5. 서브셋 데이터로 1차 검증 (엄청난 속도 향상)
        # S_cands.T -> (64, V)
        # X_sub @ S_cands.T -> (5000, V)
        preds_sub = torch.matmul(X_sub, S_cands.T) % 2 
        corrects_sub = (preds_sub == y_sub.unsqueeze(1)).sum(dim=0).float()
        acc_sub = corrects_sub / TEST_SZ
        
        max_idx = acc_sub.argmax()
        max_acc = acc_sub[max_idx].item()
        
        print(f"Batch {it:2d} | Valid Matrices: {V:,} | Max Sub-Accuracy: {max_acc:.4f} [{(time.time()-it_start):.2f}s]")
        
        # 6. 서브셋 일치율이 70% 이상이면 (노이즈 20% 이므로 정상 신호 포착) 전체 데이터 검증
        if max_acc > 0.70:
            print(f"\n[!] 유력 후보 발견 (Sub-Acc: {max_acc:.4f})! 전체 데이터 검증 진입...")
            best_cand = S_cands[max_idx].unsqueeze(1) # (64, 1)
            
            preds_all = torch.matmul(X_all_f, best_cand).squeeze(1) % 2
            acc_all = (preds_all == y_all_f).float().mean().item()
            
            print(f"전체 잔차율 (Accuracy): {acc_all:.6f}")
            
            if acc_all > 0.75:
                # 완벽한 정답 추출
                ans_vector = S_cands[max_idx].to(torch.int32).cpu().numpy()
                offsets = [i + 1 for i in range(W) if ans_vector[i] == 1]
                
                print("\n" + "="*60)
                print(f"정답 발견 (Random Subset LPN) !!")
                print(f"오프셋(S): {offsets}")
                print(f"파라미터수(W): {max(offsets)}")
                print(f"정확도(Rate): {acc_all:.6f}")
                
                answer_str = generate_answer(offsets, TEST_PREFIX)
                
                print(f"\n{'='*60}")
                print("[최종 산출물 - 규격 확인]")
                print(f"필드 : answer")
                print(f"타입 : string")
                print(f"길이 : {len(answer_str)} bit (요구사항: 192 bit)")
                print(f"{'-'*60}")
                print(answer_str)
                print(f"{'='*60}")
                
                print(f"총 소요시간: {time.time()-t0:.2f} s")
                return

    print("탐색 종료. 정답을 찾지 못했습니다.")

if __name__ == '__main__':
    solve_lpn_random_subset()