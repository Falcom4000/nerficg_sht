#!/usr/bin/env bash
# benchmark.sh — unified benchmark runner for GaussianSplatting / FasterGS / FasterGSFused / FasterGSFusedfast / FasterGSDash / FasterGSFusedDash
#
# Usage:
#   bash scripts/benchmark.sh [--methods m1,m2,...] [--scenes s1,s2,...] [--repeats N]
#
# Methods: gaussiansplatting | fastergs | fastergsfused | fastergsfusedfast | fastergsdash | fastergsfuseddash  (default: all)
# Scenes:  bonsai | counter | kitchen | room | bicycle | garden | stump  (default: bonsai)
# Repeats: integer ≥ 1 (default: 1)
#
# Examples:
#   bash scripts/benchmark.sh
#       → all methods × bonsai × 1 run
#
#   bash scripts/benchmark.sh --methods fastergsdash,fastergsfuseddash --scenes bonsai,garden,bicycle --repeats 3
#       → 2 methods × 3 scenes × 3 repeats = 18 runs
#
#   bash scripts/benchmark.sh --methods fastergs --scenes bonsai,counter,kitchen,room,bicycle,garden,stump
#       → full 7-scene sweep for FasterGS
#
# Output: experiments/benchmark_results.md  (rows appended on each run)

set -euo pipefail

CONDA_ENV="nerficg_colmap"
RESULTS_FILE="experiments/benchmark_results.md"

# ── method metadata ───────────────────────────────────────────────────────────
# config dir and output dir prefix for each method key
declare -A METHOD_CONFIG_DIR=(
    [gaussiansplatting]="configs/benchmark/gaussiansplatting"
    [fastergs]="configs/benchmark/fastergs"
    [fastergsfused]="configs/benchmark/fastergsfused"
    [fastergsfusedfast]="configs/benchmark/fastergsfusedfast"
    [fastergsdash]="configs/benchmark/fastergsdash"
    [fastergsfuseddash]="configs/benchmark/fastergsfuseddash"
)
declare -A METHOD_OUTPUT_DIR=(
    [gaussiansplatting]="output/GaussianSplatting"
    [fastergs]="output/FasterGS"
    [fastergsfused]="output/FasterGSFused"
    [fastergsfusedfast]="output/FasterGSFusedfast"
    [fastergsdash]="output/FasterGSDash"
    [fastergsfuseddash]="output/FasterGSFusedDash"
)
declare -A METHOD_DISPLAY=(
    [gaussiansplatting]="GaussianSplatting"
    [fastergs]="FasterGS"
    [fastergsfused]="FasterGSFused"
    [fastergsfusedfast]="FasterGSFusedfast"
    [fastergsdash]="FasterGSDash"
    [fastergsfuseddash]="FasterGSFusedDash"
)

ALL_METHODS=(gaussiansplatting fastergs fastergsfused fastergsfusedfast fastergsdash fastergsfuseddash)
ALL_SCENES=(bonsai counter kitchen room bicycle garden stump)

# ── defaults ──────────────────────────────────────────────────────────────────
METHODS=("${ALL_METHODS[@]}")
SCENES=(bonsai)
REPEATS=1
BENCH_VERSION="—"
BENCH_COMMIT=""

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)
            IFS=',' read -ra METHODS <<< "$2"; shift 2 ;;
        --scenes)
            IFS=',' read -ra SCENES <<< "$2"; shift 2 ;;
        --repeats)
            REPEATS="$2"; shift 2 ;;
        --version)
            BENCH_VERSION="$2"; shift 2 ;;
        --commit)
            BENCH_COMMIT="$2"; shift 2 ;;
        -h|--help)
            head -20 "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# validate methods
for m in "${METHODS[@]}"; do
    if [[ -z "${METHOD_CONFIG_DIR[$m]+x}" ]]; then
        echo "Unknown method: $m  (valid: ${ALL_METHODS[*]})" >&2; exit 1
    fi
done

# ── setup ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p experiments

if [[ ! -f "$RESULTS_FILE" ]]; then
    cat > "$RESULTS_FILE" << 'HEADER'
# Benchmark Results

| Date | Version | Commit | Method | Scene | Run | PSNR | SSIM | LPIPS | Time (s) | VRAM alloc (GiB) | VRAM resv (GiB) | Gaussians |
|------|---------|--------|--------|-------|-----|------|------|-------|----------|-----------------|-----------------|-----------|
HEADER
fi

# ── helpers ───────────────────────────────────────────────────────────────────
latest_output_dir() {
    local method_outdir="$1" scene="$2"
    ls -dt "${method_outdir}/${scene}_"* 2>/dev/null | head -1
}

parse_vram_file() {
    local f="$1"
    local alloc reserved
    alloc=$(grep -oP 'VRAM_allocated:\K[0-9]+' "$f" 2>/dev/null || echo "")
    reserved=$(grep -oP 'VRAM_reserved:\K[0-9]+' "$f" 2>/dev/null || echo "")
    if [[ -n "$alloc" && -n "$reserved" ]]; then
        awk "BEGIN{printf \"%.2f %.2f\", $alloc/1073741824, $reserved/1073741824}"
    else
        echo "? ?"
    fi
}

parse_vram_log() {
    local log="$1"
    local line alloc reserved
    line=$(grep -oP 'peak VRAM usage during training: \K[^\n]+' "$log" 2>/dev/null | tail -1 || true)
    if [[ -n "$line" ]]; then
        alloc=$(echo "$line" | grep -oP '[\d.]+(?= GiB allocated)' || echo "?")
        reserved=$(echo "$line" | grep -oP '[\d.]+(?= GiB reserved)' || echo "?")
        echo "$alloc $reserved"
    else
        echo "? ?"
    fi
}

# ── build run list ────────────────────────────────────────────────────────────
declare -a RUN_METHODS RUN_SCENES RUN_IDXS
for method in "${METHODS[@]}"; do
    for scene in "${SCENES[@]}"; do
        for ((r=1; r<=REPEATS; r++)); do
            RUN_METHODS+=("$method")
            RUN_SCENES+=("$scene")
            RUN_IDXS+=("$r")
        done
    done
done

TOTAL=${#RUN_METHODS[@]}
echo "═══════════════════════════════════════════════════════════════"
printf "  Benchmark — %d runs  (%s)\n" "$TOTAL" "$(date '+%Y-%m-%d %H:%M')"
echo "  Methods : ${METHODS[*]}"
echo "  Scenes  : ${SCENES[*]}"
echo "  Repeats : $REPEATS"
echo "  Results → $RESULTS_FILE"
echo "═══════════════════════════════════════════════════════════════"

# ── main loop ─────────────────────────────────────────────────────────────────
for ((i=0; i<TOTAL; i++)); do
    method="${RUN_METHODS[$i]}"
    scene="${RUN_SCENES[$i]}"
    rep="${RUN_IDXS[$i]}"
    run_num=$((i+1))

    config_dir="${METHOD_CONFIG_DIR[$method]}"
    output_base="${METHOD_OUTPUT_DIR[$method]}"
    display="${METHOD_DISPLAY[$method]}"
    config="${config_dir}/${scene}.yaml"

    echo ""
    echo "── Run ${run_num}/${TOTAL}: ${display} / ${scene} (rep ${rep}) ──────────────"
    echo "   Config: ${config}"

    if [[ ! -f "$config" ]]; then
        echo "   ⚠  Config not found — skipping"
        continue
    fi

    LOG_FILE=$(mktemp "/tmp/bench_${method}_${scene}_XXXXXX.log")
    T_START=$(date +%s)

    conda run -n "$CONDA_ENV" python scripts/train.py \
        -c "$config" \
        TRAINING.WRITE_VRAM_STATS=true \
        2>&1 | tee "$LOG_FILE"

    T_END=$(date +%s)
    ELAPSED=$((T_END - T_START))

    # locate output dir
    OUT_DIR=$(latest_output_dir "$output_base" "$scene")
    if [[ -z "$OUT_DIR" ]]; then
        echo "   ⚠  Output directory not found"
        rm -f "$LOG_FILE"; continue
    fi
    echo "   Output: $OUT_DIR"

    # VRAM
    VRAM_ALLOC="?"; VRAM_RESV="?"
    VRAM_FILE="${OUT_DIR}/vram_stats.txt"
    if [[ -f "$VRAM_FILE" ]]; then
        read -r VRAM_ALLOC VRAM_RESV <<< "$(parse_vram_file "$VRAM_FILE")"
    else
        read -r VRAM_ALLOC VRAM_RESV <<< "$(parse_vram_log "$LOG_FILE")"
    fi

    # Gaussians
    N_GAUSS="?"
    N_GAUSS_FILE="${OUT_DIR}/n_gaussians.txt"
    [[ -f "$N_GAUSS_FILE" ]] && N_GAUSS=$(grep -oP 'N_Gaussians:\K[0-9]+' "$N_GAUSS_FILE" 2>/dev/null || echo "?")

    # metrics
    PSNR="?"; SSIM="?"; LPIPS="?"
    METRICS="${OUT_DIR}/test_30000/metrics_8bit.txt"
    if [[ -f "$METRICS" ]]; then
        LAST=$(tail -1 "$METRICS")
        PSNR=$(echo "$LAST"  | grep -oP 'PSNR:\K[\d.]+' || echo "?")
        SSIM=$(echo "$LAST"  | grep -oP 'SSIM:\K[\d.]+' || echo "?")
        LPIPS=$(echo "$LAST" | grep -oP 'LPIPS:\K[\d.]+' || echo "?")
    fi

    rm -f "$LOG_FILE"

    printf "   Time: %ds | VRAM: %s/%s GiB | Gaussians: %s\n" \
        "$ELAPSED" "$VRAM_ALLOC" "$VRAM_RESV" "$N_GAUSS"
    printf "   PSNR: %s | SSIM: %s | LPIPS: %s\n" "$PSNR" "$SSIM" "$LPIPS"

    # append to results
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
    GIT_COMMIT="${BENCH_COMMIT:-$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "?")}"
    printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
        "$TIMESTAMP" "$BENCH_VERSION" "$GIT_COMMIT" "$display" "$scene" "$rep" \
        "$PSNR" "$SSIM" "$LPIPS" \
        "$ELAPSED" "$VRAM_ALLOC" "$VRAM_RESV" "$N_GAUSS" \
        >> "$RESULTS_FILE"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All runs complete."
echo "═══════════════════════════════════════════════════════════════"
echo ""
cat "$RESULTS_FILE"
