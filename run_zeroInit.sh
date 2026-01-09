#!/bin/bash

# ============================================================================
# 零初始化改进消融实验
# 测试每个改进模块的独立效果和组合效果
# ============================================================================

data_dir="/home/jiangkaini/Dataset/university1652/train"
test_dir="/home/jiangkaini/Dataset/university1652/test"
gpu_ids="0"
batchsize=16
lr=0.01
epochs=200
triplet_loss=0.3
model="convnext_tiny"

echo "════════════════════════════════════════════════════════════════════════"
echo "Zero-Initialization Ablation Study"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Data:       $data_dir"
echo "  - Batch size: $batchsize"
echo "  - LR:         $lr"
echo "  - Epochs:     $epochs"
echo "  - GPU:        $gpu_ids"
echo ""
echo "Experiments:"
echo "  - Exp 0: Baseline (Original MCCG)"
echo "  - Exp 1: +Zero-Init TripletAttention"
echo "  - Exp 2: +Zero-Init DetailBranch"
echo "  - Exp 3: +Zero-Init AFF"
echo "  - Exp 4: +Consistency Regularization"
echo "  - Exp 5: +Progressive Training"
echo "  - Exp 6: All Improvements (Best)"
echo ""
echo "Estimated time: ~31.5 hours (7 experiments × 4.5 hours)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# 记录开始时间
start_time=$(date +%s)

# ============================================================================
# Exp 0: Baseline (Original MCCG)
# ============================================================================

# echo ""
# echo "════════════════════════════════════════════════════════════════════════"
# echo "Exp 0: Baseline (Original MCCG)"
# echo "════════════════════════════════════════════════════════════════════════"

# if [ -d "./model/zero_init_exp0_baseline" ] && [ -f "./model/zero_init_exp0_baseline/net_199.pth" ]; then
#     echo "✅ Baseline model already exists, skipping training..."
# else
#     python train2.py \
#         --name zero_init_exp0_baseline \
#         --data_dir $data_dir \
#         --batchsize $batchsize \
#         --lr $lr \
#         --triplet_loss $triplet_loss \
#         --epochs $epochs \
#         --gpu_ids $gpu_ids \
#         --model $model
# fi

# # 测试
# python test.py --name zero_init_exp0_baseline --test_dir $test_dir --mode 1 --gpu_ids $gpu_ids
# python test.py --name zero_init_exp0_baseline --test_dir $test_dir --mode 2 --gpu_ids $gpu_ids

# # ============================================================================
# # Exp 1: +Zero-Init TripletAttention Only
# # ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "Exp 1: +Zero-Init TripletAttention Only"
echo "════════════════════════════════════════════════════════════════════════"

python train2.py \
    --name zero_init_exp1_tri \
    --data_dir $data_dir \
    --use_zero_init \
    --use_zero_init_tri \
    --batchsize $batchsize \
    --lr $lr \
    --triplet_loss $triplet_loss \
    --epochs $epochs \
    --gpu_ids $gpu_ids

python test.py --name zero_init_exp1_tri --test_dir $test_dir --mode 1 --gpu_ids $gpu_ids
python test.py --name zero_init_exp1_tri --test_dir $test_dir --mode 2 --gpu_ids $gpu_ids

# ============================================================================
# Exp 2: +Zero-Init DetailBranch Only
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "Exp 2: +Zero-Init DetailBranch Only"
echo "════════════════════════════════════════════════════════════════════════"

python train2.py \
    --name zero_init_exp2_detail \
    --data_dir $data_dir \
    --use_zero_init \
    --use_zero_init_detail \
    --batchsize $batchsize \
    --lr $lr \
    --triplet_loss $triplet_loss \
    --epochs $epochs \
    --gpu_ids $gpu_ids

python test.py --name zero_init_exp2_detail --test_dir $test_dir --mode 1 --gpu_ids $gpu_ids
python test.py --name zero_init_exp2_detail --test_dir $test_dir --mode 2 --gpu_ids $gpu_ids

# ============================================================================
# Exp 3: +Zero-Init AFF Only
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "Exp 3: +Zero-Init AFF Only"
echo "════════════════════════════════════════════════════════════════════════"

python train2.py \
    --name zero_init_exp3_aff \
    --data_dir $data_dir \
    --use_zero_init \
    --use_zero_init_aff \
    --batchsize $batchsize \
    --lr $lr \
    --triplet_loss $triplet_loss \
    --epochs $epochs \
    --gpu_ids $gpu_ids

python test.py --name zero_init_exp3_aff --test_dir $test_dir --mode 1 --gpu_ids $gpu_ids
python test.py --name zero_init_exp3_aff --test_dir $test_dir --mode 2 --gpu_ids $gpu_ids

# ============================================================================
# Exp 4: +Consistency Regularization (with Zero-Init Tri)
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "Exp 4: +Consistency Regularization"
echo "════════════════════════════════════════════════════════════════════════"

python train2.py \
    --name zero_init_exp4_consistency \
    --data_dir $data_dir \
    --use_zero_init \
    --use_zero_init_tri \
    --consistency_weight 0.1 \
    --batchsize $batchsize \
    --lr $lr \
    --triplet_loss $triplet_loss \
    --epochs $epochs \
    --gpu_ids $gpu_ids

python test.py --name zero_init_exp4_consistency --test_dir $test_dir --mode 1 --gpu_ids $gpu_ids
python test.py --name zero_init_exp4_consistency --test_dir $test_dir --mode 2 --gpu_ids $gpu_ids

# ============================================================================
# Exp 5: +Progressive Training (with Zero-Init Tri)
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "Exp 5: +Progressive Training"
echo "════════════════════════════════════════════════════════════════════════"

python train2.py \
    --name zero_init_exp5_progressive \
    --data_dir $data_dir \
    --use_zero_init \
    --use_zero_init_tri \
    --progressive_training \
    --freeze_epochs 50 \
    --batchsize $batchsize \
    --lr $lr \
    --triplet_loss $triplet_loss \
    --epochs $epochs \
    --gpu_ids $gpu_ids

python test.py --name zero_init_exp5_progressive --test_dir $test_dir --mode 1 --gpu_ids $gpu_ids
python test.py --name zero_init_exp5_progressive --test_dir $test_dir --mode 2 --gpu_ids $gpu_ids

# ============================================================================
# Exp 6: All Improvements (Best Configuration)
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "Exp 6: All Improvements (Best Configuration)"
echo "════════════════════════════════════════════════════════════════════════"

python train2.py \
    --name zero_init_exp6_all \
    --data_dir $data_dir \
    --use_zero_init \
    --use_zero_init_tri \
    --use_zero_init_detail \
    --use_zero_init_aff \
    --consistency_weight 0.1 \
    --progressive_training \
    --freeze_epochs 50 \
    --batchsize $batchsize \
    --lr $lr \
    --triplet_loss $triplet_loss \
    --epochs $epochs \
    --gpu_ids $gpu_ids

python test.py --name zero_init_exp6_all --test_dir $test_dir --mode 1 --gpu_ids $gpu_ids
python test.py --name zero_init_exp6_all --test_dir $test_dir --mode 2 --gpu_ids $gpu_ids

# ============================================================================
# 计算总耗时
# ============================================================================

end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "All Experiments Completed!"
echo "════════════════════════════════════════════════════════════════════════"
echo "Total time: ${hours}h ${minutes}m"
echo ""

# ============================================================================
# 生成结果汇总
# ============================================================================

echo "════════════════════════════════════════════════════════════════════════"
echo "Zero-Initialization Ablation Summary"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

extract_results() {
    local model_name=$1
    local result_file="./model/$model_name/result.txt"
    
    if [ -f "$result_file" ]; then
        local mode2=$(grep "Recall@1" "$result_file" | head -1 | awk '{print $2}')
        local mode1=$(grep "Recall@1" "$result_file" | tail -1 | awk '{print $2}')
        echo "  Mode 2 (D→S): ${mode2}    Mode 1 (S→D): ${mode1}"
    else
        echo "  ⚠️  Result file not found"
    fi
}

echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ Exp 0: Baseline (Original MCCG)                                        │"
echo "└────────────────────────────────────────────────────────────────────────┘"
extract_results "zero_init_exp0_baseline"

echo ""
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ Exp 1: +Zero-Init TripletAttention                                     │"
echo "└────────────────────────────────────────────────────────────────────────┘"
extract_results "zero_init_exp1_tri"

echo ""
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ Exp 2: +Zero-Init DetailBranch                                         │"
echo "└────────────────────────────────────────────────────────────────────────┘"
extract_results "zero_init_exp2_detail"

echo ""
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ Exp 3: +Zero-Init AFF                                                  │"
echo "└────────────────────────────────────────────────────────────────────────┘"
extract_results "zero_init_exp3_aff"

echo ""
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ Exp 4: +Consistency Regularization                                     │"
echo "└────────────────────────────────────────────────────────────────────────┘"
extract_results "zero_init_exp4_consistency"

echo ""
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ Exp 5: +Progressive Training                                           │"
echo "└────────────────────────────────────────────────────────────────────────┘"
extract_results "zero_init_exp5_progressive"

echo ""
echo "┌────────────────────────────────────────────────────────────────────────┐"
echo "│ Exp 6: All Improvements (Best)                                         │"
echo "└────────────────────────────────────────────────────────────────────────┘"
extract_results "zero_init_exp6_all"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "📊 Detailed Results"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved in:"
for exp in exp0_baseline exp1_tri exp2_detail exp3_aff exp4_consistency exp5_progressive exp6_all; do
    echo "  - ./model/zero_init_${exp}/result.txt"
done
echo ""

# ============================================================================
# 生成对比表格 CSV
# ============================================================================

echo "Generating comparison CSV..."

cat > zero_init_ablation_results.csv << EOF
Experiment,Module,Mode2_D2S,Mode1_S2D,Improvement
EOF

# 提取并写入 CSV
for exp_name in "exp0_baseline" "exp1_tri" "exp2_detail" "exp3_aff" "exp4_consistency" "exp5_progressive" "exp6_all"; do
    result_file="./model/zero_init_${exp_name}/result.txt"
    
    if [ -f "$result_file" ]; then
        mode2=$(grep "Recall@1" "$result_file" | head -1 | awk '{print $2}')
        mode1=$(grep "Recall@1" "$result_file" | tail -1 | awk '{print $2}')
        
        # 计算提升（相对 baseline）
        if [ "$exp_name" = "exp0_baseline" ]; then
            baseline_mode2=$mode2
            improvement="0.0"
        else
            improvement=$(echo "$mode2 - $baseline_mode2" | bc)
        fi
        
        # 实验描述
        case $exp_name in
            "exp0_baseline") desc="Baseline" ;;
            "exp1_tri") desc="Zero-Init Tri" ;;
            "exp2_detail") desc="Zero-Init Detail" ;;
            "exp3_aff") desc="Zero-Init AFF" ;;
            "exp4_consistency") desc="Consistency Reg" ;;
            "exp5_progressive") desc="Progressive Train" ;;
            "exp6_all") desc="All Improvements" ;;
        esac
        
        echo "$desc,$exp_name,$mode2,$mode1,+$improvement" >> zero_init_ablation_results.csv
    fi
done

echo "✅ CSV saved to: zero_init_ablation_results.csv"
echo ""

