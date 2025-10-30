#!/usr/bin/env fish
# 快速测试版 Pipeline - 用于调试和快速迭代
# 
# 特点:
#   • 更少的 epochs
#   • 更小的 batch size
#   • 更少的采样数量
#   • 适合快速验证代码改动
#
# 使用方法:
#   ./quick_test.fish          # 测试所有 Phase
#   ./quick_test.fish 2 3 4    # 跳过数据准备，只测试模型训练

if test (count $argv) -eq 0
    set phases 1 2 3 4
else
    set phases $argv
end

echo "🧪 AMP-FlowRAE 快速测试模式"
echo "🎯 Testing Phases: $phases"
echo ""

function run_step
    set step_name $argv[1]
    set cmd $argv[2..-1]
    echo "🔸 $step_name"
    eval $cmd
    or begin
        echo "❌ 失败: $step_name"
        exit 1
    end
end

function phase_header
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo $argv[1]
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
end

# ============================================================================
# Phase 1: 数据准备 (与正式版相同)
# ============================================================================

if contains 1 $phases
    phase_header "📊 Phase 1: 数据准备"
    
    run_step "解析 FASTA" python scripts/01_parse_fasta.py
    run_step "去重" python scripts/02_deduplicate.py
    run_step "提取 embeddings" python scripts/03_extract_esm3.py
    run_step "划分数据" python scripts/04_split_data.py
    run_step "计算标签" python scripts/05_compute_labels.py
    run_step "归一化" python scripts/06_normalize_embeddings.py
    
    echo "✅ Phase 1 完成"
end

# ============================================================================
# Phase 2: RAE 快速训练 (10 epochs)
# ============================================================================

if contains 2 $phases
    phase_header "🧠 Phase 2: RAE (快速模式)"
    
    run_step "训练 RAE (10 epochs)" \
        python rae/train.py \
            --epochs 10 \
            --batch-size 128 \
            --lr 1e-3 \
            --output-dir outputs/rae_test
    
    run_step "评估 RAE" \
        python rae/evaluate.py \
            --checkpoint outputs/rae_test/checkpoints/best.pt \
            --output-dir outputs/rae_test_eval
    
    echo "✅ Phase 2 完成"
end

# ============================================================================
# Phase 3: Flow Matching 快速训练 (20 epochs)
# ============================================================================

if contains 3 $phases
    phase_header "🌊 Phase 3: Flow (快速模式)"
    
    run_step "训练 Flow (20 epochs)" \
        python flow_matching/train.py \
            --epochs 20 \
            --batch-size 256 \
            --lr 1e-4 \
            --output-dir outputs/flow_test
    
    run_step "采样 (100 samples)" \
        python flow_matching/sample.py \
            --checkpoint outputs/flow_test/checkpoints/best.pt \
            --num-samples 100 \
            --ode-steps 50 \
            --mode random \
            --use-data-range \
            --output-dir outputs/samples_test
    
    run_step "评估采样" \
        python flow_matching/evaluate_samples.py \
            --samples-dir outputs/samples_test
    
    echo "✅ Phase 3 完成"
end

# ============================================================================
# Phase 4: 解码与分析
# ============================================================================

if contains 4 $phases
    phase_header "🔬 Phase 4: 解码与分析"
    
    run_step "解码序列" \
        python decoder/decode.py \
            --latents outputs/samples_test/sampled_latents.npy \
            --conditions outputs/samples_test/sampled_conditions.csv \
            --rae-checkpoint outputs/rae_test/checkpoints/best.pt \
            --output outputs/decoded_sequences_test.csv
    
    run_step "分析序列" \
        python scripts/09_analyze_generated.py \
            --input outputs/decoded_sequences_test.csv \
            --output-dir outputs/analysis_test
    
    echo "✅ Phase 4 完成"
end

echo ""
phase_header "🎉 测试完成！"
echo ""
echo "📁 测试输出:"
echo "  • RAE:      outputs/rae_test/"
echo "  • Flow:     outputs/flow_test/"
echo "  • 序列:     outputs/decoded_sequences_test.csv"
echo "  • 分析:     outputs/analysis_test/"
echo ""
echo "💡 提示:"
echo "  如果测试通过，运行完整版本:"
echo "  ./run_pipeline.fish"
echo ""


