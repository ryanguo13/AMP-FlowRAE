#!/usr/bin/env fish
# AMP-FlowRAE Pipeline Runner - 灵活版本
# 
# 使用方法:
#   ./run_pipeline.fish                    # 运行所有 Phase
#   ./run_pipeline.fish 1 2                # 只运行 Phase 1 和 2
#   ./run_pipeline.fish 3                  # 只运行 Phase 3
#   ./run_pipeline.fish 4                  # 只运行 Phase 4
#
# Phase 说明:
#   1 - 数据准备 (解析, 去重, embedding 提取)
#   2 - RAE 训练 (潜空间学习)
#   3 - Flow Matching (条件生成)
#   4 - 解码与分析 (生成序列)

# 参数解析
if test (count $argv) -eq 0
    # 没有参数，运行所有 Phase
    set phases 1 2 3 4
else
    # 使用用户指定的 Phase
    set phases $argv
end

set -g PROJECT_ROOT (pwd)
echo "🚀 AMP-FlowRAE Pipeline"
echo "📂 Root: $PROJECT_ROOT"
echo "🎯 Running Phases: $phases"
echo ""

# 检查虚拟环境
if not test -d .venv
    echo "❌ 虚拟环境不存在，请先运行: fish scripts/setup_env.fish"
    exit 1
end

# ============================================================================
# 辅助函数
# ============================================================================

function run_step
    set step_name $argv[1]
    set cmd $argv[2..-1]
    
    echo "🔸 $step_name"
    eval $cmd
    if test $status -ne 0
        echo "❌ 失败: $step_name"
        exit 1
    end
end

function phase_header
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo $argv[1]
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
end

# ============================================================================
# Phase 1: 数据准备
# ============================================================================

if contains 1 $phases
    phase_header "📊 Phase 1: 数据准备"
    
    run_step "解析 FASTA" python scripts/01_parse_fasta.py
    run_step "序列去重" python scripts/02_deduplicate.py
    run_step "提取 ESM-3 embeddings" python scripts/03_extract_esm3.py
    run_step "划分数据集" python scripts/04_split_data.py
    run_step "计算标签" python scripts/05_compute_labels.py
    run_step "归一化 embeddings" python scripts/06_normalize_embeddings.py
    run_step "验证数据" python scripts/validate_data.py
    
    echo "✅ Phase 1 完成"
end

# ============================================================================
# Phase 2: RAE 潜空间学习
# ============================================================================

if contains 2 $phases
    phase_header "🧠 Phase 2: RAE 潜空间学习"
    
    run_step "训练 RAE" \
        python rae/train.py \
            --epochs 500 \
            --batch-size 256 \
            --lr 1e-3 \
            --output-dir outputs/rae
    
    run_step "评估 RAE" \
        python rae/evaluate.py \
            --checkpoint outputs/rae/checkpoints/best.pt \
            --output-dir outputs/rae_eval
    
    echo "✅ Phase 2 完成"
end

# ============================================================================
# Phase 3: Conditional Flow Matching
# ============================================================================

if contains 3 $phases
    phase_header "🌊 Phase 3: Conditional Flow Matching"
    
    run_step "训练 Flow Matching" \
        python flow_matching/train.py \
            --epochs 500 \
            --batch-size 512 \
            --lr 1e-4 \
            --output-dir outputs/flow
    
    run_step "采样 latent vectors" \
        python flow_matching/sample.py \
            --checkpoint outputs/flow/checkpoints/best.pt \
            --num-samples 500 \
            --ode-steps 100 \
            --mode grid \
            --use-data-range \
            --output-dir outputs/samples
    
    run_step "评估采样质量" \
        python flow_matching/evaluate_samples.py \
            --samples-dir outputs/samples
    
    echo "✅ Phase 3 完成"
end

# ============================================================================
# Phase 4: 序列解码与分析
# ============================================================================

if contains 4 $phases
    phase_header "🔬 Phase 4: 序列解码与分析"
    
    run_step "解码序列" \
        python decoder/decode.py \
            --latents outputs/samples/sampled_latents.npy \
            --conditions outputs/samples/sampled_conditions.csv \
            --rae-checkpoint outputs/rae/checkpoints/best.pt \
            --output outputs/decoded_sequences.csv
    
    run_step "分析生成序列" \
        python scripts/09_analyze_generated.py \
            --input outputs/decoded_sequences.csv \
            --output-dir outputs/analysis
    
    echo "✅ Phase 4 完成"
end

# ============================================================================
# 完成总结
# ============================================================================

echo ""
phase_header "🎉 完成！"
echo ""
echo "已完成 Phases: $phases"
echo ""
echo "📁 输出目录:"
echo "  • 数据:        data/embeddings/, data/splits/"
echo "  • RAE:         outputs/rae/"
echo "  • Flow:        outputs/flow/"
echo "  • 生成序列:    outputs/decoded_sequences.csv"
echo "  • 分析:        outputs/analysis/"
echo ""
echo "📊 快速查看:"
echo "  • TensorBoard: tensorboard --logdir outputs/"
echo "  • 可视化:      open outputs/analysis/*.png"
echo ""

