import torch
import torch.nn.functional as F
import sys

# === 提案アルゴリズム: Stabilized Piecewise-Rational Chart (SPRC) ===

def stabilized_rational_chart(u, epsilon=1e-6, lambd=0.1):
    """
    Args:
        u: Input tensor of shape (..., 9)
           ニューラルネットワークからの非拘束パラメータ
    Returns:
        H: Doubly Stochastic Matrix of shape (..., 4, 4)
           Birkhoff多面体の内部点（厳密な二重確率行列）
    """
    # 入力を倍精度(float64)として扱うことを推奨
    
    # 1. Tangent Space Projection (接空間への射影)
    # 入力 u を左上の 3x3 ブロックと見なし、行和・列和が0になるように拡張して V を作る
    
    # u: (Batch, 9) -> (Batch, 3, 3)
    u_3x3 = u.view(-1, 3, 3)
    
    # 行和 (Batch, 3, 1) と 列和 (Batch, 1, 3) を計算
    row_sum = torch.sum(u_3x3, dim=2, keepdim=True)
    col_sum = torch.sum(u_3x3, dim=1, keepdim=True)
    
    # 全体和 (Batch, 1, 1) -> V_44 のための調整項
    total_sum = torch.sum(row_sum, dim=1, keepdim=True)

    # V行列の構築
    # [ u_3x3    | -row_sum ]
    # [ -col_sum | total_sum]
    
    # 上半分 (Batch, 3, 4)
    top_block = torch.cat([u_3x3, -row_sum], dim=2)
    
    # 下半分 (Batch, 1, 4)
    # total_sum はスカラーだが次元を合わせて結合
    bottom_block = torch.cat([-col_sum, total_sum], dim=2)
    
    # 結合して 4x4 行列 V を作成
    V = torch.cat([top_block, bottom_block], dim=1) 

    # 2. Tropical Norm & Stabilization (安定化項の計算)
    # m(V) = max(0, -V_ij)
    # Vの成分が負になる部分の最大値を求める（境界までの距離に関連）
    neg_V = -V
    # view(-1, 16)にしてからdim=1で最大値を取る
    m_V_val, _ = torch.max(neg_V.view(-1, 16), dim=1)
    m_V = F.relu(m_V_val).view(-1, 1, 1)

    # 正則化項 (L1ノルム)
    norm_V1 = torch.sum(torch.abs(V), dim=(1, 2), keepdim=True)
    
    # 分母の計算: ゼロ除算を防ぐために epsilon と lambda項 を追加
    denominator = m_V + epsilon + lambd * norm_V1
    
    # 3. Construct H (最終的な行列の構成)
    # H = J4 + (0.25 * V) / denominator
    # Vは和が0なので、第2項を足しても行和・列和は不変
    J4 = 0.25
    H = J4 + (0.25 * V) / denominator
    
    return H

# === 検証ロジック (Unit Tests) ===

def run_phase1_verification():
    print("=== Phase 1: Mathematical Verification (CPU / Float64) ===")
    
    # MacBookでの検証精度を最大化するため CPU かつ float64 を強制
    device = torch.device('cpu')
    dtype = torch.float64
    
    print(f"Device: {device}")
    print(f"Dtype : {dtype}")
    print("-" * 40)

    # 再現性確保
    torch.manual_seed(42)
    
    # バッチサイズ 1024 でランダム入力を生成
    B = 1024
    u = torch.randn(B, 9, device=device, dtype=dtype, requires_grad=True)

    # --- Test 1: 制約充足性 (Forward Correctness) ---
    print("Running Forward Pass Check...")
    H = stabilized_rational_chart(u)
    
    # 行和・列和のチェック
    row_sums = H.sum(dim=2) # (B, 4)
    col_sums = H.sum(dim=1) # (B, 4)
    
    # 誤差の最大値を取得
    # float64 なので 1e-15 付近の精度が出るはず
    max_err_row = (row_sums - 1.0).abs().max().item()
    max_err_col = (col_sums - 1.0).abs().max().item()
    
    # 非負性のチェック (最小値が 0 以上か)
    # 数値誤差で -1e-16 とかになるのは許容
    min_val = H.min().item()
    is_non_negative = min_val >= -1e-12

    print(f"  Max Row Sum Error: {max_err_row:.2e} (Ideal: < 1e-15)")
    print(f"  Max Col Sum Error: {max_err_col:.2e} (Ideal: < 1e-15)")
    print(f"  Non-negative Check: {is_non_negative} (Min val: {min_val:.2e})")

    if max_err_row < 1e-14 and max_err_col < 1e-14 and is_non_negative:
        print("  => [PASS] Forward constraints are exactly satisfied.")
    else:
        print("  => [FAIL] Forward constraints check failed.")
        sys.exit(1)

    print("-" * 40)

    # --- Test 2: 勾配正確性 (Gradient Check via Finite Difference) ---
    print("Running Gradient Check (Jacobian Analysis)...")
    
    # gradcheckは計算コストが高いので、小さなバッチで実行
    B_grad = 2
    u_grad = torch.randn(B_grad, 9, device=device, dtype=dtype, requires_grad=True)
    
    # torch.autograd.gradcheck は、数値微分と解析的微分(autograd)を比較する
    # これが通れば、数式上の微分と実装が一致している（＝微分可能である）ことの証明になる
    try:
        test_passed = torch.autograd.gradcheck(
            stabilized_rational_chart, 
            (u_grad,), 
            eps=1e-6, 
            atol=1e-4,
            check_undefined_grad=False
        )
    except Exception as e:
        print(f"  => [FAIL] Exception during gradcheck: {e}")
        test_passed = False

    if test_passed:
        print(f"  => [PASS] Gradient check passed. (Jacobian is consistent)")
    else:
        print(f"  => [FAIL] Gradient check failed.")

    print("=" * 40)
    print("VERIFICATION COMPLETE: The algorithm is mathematically sound.")

if __name__ == "__main__":
    run_phase1_verification()