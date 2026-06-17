#!/usr/bin/env bash
# mkmockproj.sh <target_dir> [base_lat_us] [profile]
#
# Generates a fake kernel project (its own git repo) for WarpSpeed acceptance
# tests. The "kernel" is src/kernel.cu whose magic header drives everything:
#
#     // MOCK_LAT_US: 850.0      <- measured latency of the rendered binary
#     // MOCK_PROFILE: balanced  <- which canned NCU profile the mock ncu emits
#     (optional markers anywhere in the file: MOCK_BROKEN -> correctness fails,
#      MOCK_RACY -> mock compute-sanitizer racecheck fails)
#
# ./build.sh renders bin/kernel_bench from the header (the project's
# build_command), so an "optimization" that edits MOCK_LAT_US and rebuilds
# genuinely changes what every downstream tool measures - deterministically.

set -eu

TARGET=${1:?usage: mkmockproj.sh <target_dir> [base_lat_us] [profile]}
LAT=${2:-850.0}
PROFILE=${3:-balanced}

mkdir -p "$TARGET"/src "$TARGET"/tools "$TARGET"/harness "$TARGET"/bin
cd "$TARGET"

cat > src/kernel.cu <<EOF
// Mock CUDA kernel for WarpSpeed acceptance tests.
// MOCK_LAT_US: $LAT
// MOCK_PROFILE: $PROFILE
//
// "Optimizing" this kernel = lowering MOCK_LAT_US (and optionally switching
// MOCK_PROFILE) then rebuilding. Everything downstream measures the change.

__global__ void mock_kernel_main(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}
EOF

cat > tools/render_bench.py <<'EOF'
#!/usr/bin/env python3
"""Render bin/kernel_bench from src/kernel.cu's MOCK_* header (mock 'compiler')."""
import json, re, sys

src, template, out = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src).read()

def grab(key, default):
    m = re.search(r"//\s*%s:\s*(\S+)" % key, text)
    return m.group(1) if m else default

params = {
    "base_lat_us": float(grab("MOCK_LAT_US", "1000.0")),
    "profile": grab("MOCK_PROFILE", "balanced"),
    "markers": [m for m in ("MOCK_BROKEN", "MOCK_RACY", "MOCK_LEAKY") if m in text],
}
body = open(template).read().replace("__PARAMS_JSON__", json.dumps(params))
open(out, "w").write(body)
print("rendered %s (lat=%s profile=%s markers=%s)"
      % (out, params["base_lat_us"], params["profile"], params["markers"]))
EOF

cat > tools/kernel_bench_template.py <<'EOF'
#!/usr/bin/env python3
"""Mock kernel benchmark binary (rendered by build.sh - do not edit)."""
import json, os, random, sys, time

PARAMS = __PARAMS_JSON__

DEV_OFFSETS = {0: 0.000, 1: 0.004, 2: -0.003, 3: 0.002,
               4: -0.004, 5: 0.003, 6: 0.001, 7: -0.002}
SHAPE_MULT = {"small": 0.35, "default": 1.0, "large": 2.1}

PROFILES = {
    "balanced":     dict(sm_pct=62.0, mem_pct=58.0, dram_pct=47.0, l1_pct=38.0, l2_pct=52.0,
                         l2_hit_pct=61.0, dram_gbps=1900.0, eligible_warps_per_scheduler=2.1,
                         issued_warp_per_scheduler=0.84, active_warps_per_scheduler=9.2,
                         warp_cycles_per_issued_inst=11.0, occupancy_pct=71.5,
                         theoretical_occupancy_pct=87.5, regs_per_thread=64,
                         smem_static_bytes=49152, smem_dynamic_bytes=0,
                         grid_size=264, block_size=256, waves_per_sm=2),
    "membound":     dict(sm_pct=31.0, mem_pct=88.0, dram_pct=84.0, l1_pct=42.0, l2_pct=57.0,
                         l2_hit_pct=34.0, dram_gbps=3100.0, eligible_warps_per_scheduler=1.1,
                         issued_warp_per_scheduler=0.42, active_warps_per_scheduler=11.5,
                         warp_cycles_per_issued_inst=27.5, occupancy_pct=63.0,
                         theoretical_occupancy_pct=75.0, regs_per_thread=90,
                         smem_static_bytes=32768, smem_dynamic_bytes=0,
                         grid_size=132, block_size=256, waves_per_sm=1),
    "computebound": dict(sm_pct=92.0, mem_pct=33.0, dram_pct=21.0, l1_pct=51.0, l2_pct=44.0,
                         l2_hit_pct=78.0, dram_gbps=800.0, eligible_warps_per_scheduler=3.4,
                         issued_warp_per_scheduler=0.97, active_warps_per_scheduler=8.1,
                         warp_cycles_per_issued_inst=8.2, occupancy_pct=78.0,
                         theoretical_occupancy_pct=87.5, regs_per_thread=128,
                         smem_static_bytes=16384, smem_dynamic_bytes=0,
                         grid_size=528, block_size=128, waves_per_sm=4),
}


def device():
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    try:
        return int(v)
    except ValueError:
        return 0


def lat_for(shape, rep):
    dev = device()
    base = PARAMS["base_lat_us"]
    mult = (1.0 + DEV_OFFSETS.get(dev, 0.0)) * SHAPE_MULT.get(shape, 1.0)
    rng = random.Random("%d|%s|%d|%s" % (dev, shape, rep, base))
    return base * mult * (1.0 + rng.gauss(0.0, 0.003))


def main(argv):
    reps, warmup, shape = 30, 5, "default"
    mode = "bench"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--reps":
            reps = int(argv[i + 1]); i += 2
        elif a == "--warmup":
            warmup = int(argv[i + 1]); i += 2
        elif a == "--shape":
            shape = argv[i + 1]; i += 2
        elif a == "--check":
            mode = "check"; i += 1
        elif a == "--emit-profile":
            mode = "profile"; i += 1
        else:
            i += 1

    if mode == "check":
        if "MOCK_BROKEN" in PARAMS["markers"]:
            print(json.dumps({"correct": False, "shape": shape, "detail": "mock broken marker"}))
            return 1
        print(json.dumps({"correct": True, "shape": shape}))
        return 0

    if mode == "profile":
        prof = dict(PROFILES.get(PARAMS["profile"], PROFILES["balanced"]))
        prof["duration_us"] = round(PARAMS["base_lat_us"], 1)
        prof["elapsed_cycles"] = int(PARAMS["base_lat_us"] * 1800)
        print(json.dumps({"kernels": [{"name": "mock_kernel_main", "metrics": prof}]}))
        return 0

    for r in range(warmup):
        time.sleep(lat_for(shape, -1 - r) / 1e6)
    for r in range(reps):
        lat = lat_for(shape, r)
        time.sleep(lat / 1e6)
        print("LAT_US=%.3f" % lat)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
EOF

cat > build.sh <<'EOF'
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p bin
python3 tools/render_bench.py src/kernel.cu tools/kernel_bench_template.py bin/kernel_bench
chmod +x bin/kernel_bench
EOF

cat > harness/problem_shapes.json <<'EOF'
{
  "shapes": [
    {"name": "small",   "note": "mock shape, 0.35x latency"},
    {"name": "default", "note": "mock shape, 1.0x latency"}
  ]
}
EOF

cat > harness/tolerances.json <<'EOF'
{"fp32": {"rtol": 1e-5, "atol": 1e-6}, "note": "mock harness; tolerances unused"}
EOF

cat > harness/correctness.py <<'EOF'
#!/usr/bin/env python3
"""Mock correctness harness: same CLI contract as the real WarpSpeed template.
Runs the implementation's self-check across all shapes in problem_shapes.json.
Deterministic. Exit 0 iff every shape passes; JSON report on stdout."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260612


def main(argv):
    impl, only_shape = None, None
    i = 0
    while i < len(argv):
        if argv[i] == "--impl":
            impl = argv[i + 1]; i += 2
        elif argv[i] == "--shape":
            only_shape = argv[i + 1]; i += 2
        else:
            i += 1
    if not impl:
        print(json.dumps({"ok": False, "error": "--impl required"})); return 2

    shapes = [s["name"] for s in json.load(open(os.path.join(HERE, "problem_shapes.json")))["shapes"]]
    if only_shape:
        shapes = [s for s in shapes if s == only_shape] or [only_shape]

    report = {"ok": True, "seed": SEED, "shapes": {}}
    for s in shapes:
        r = subprocess.run([impl, "--check", "--shape", s], capture_output=True, text=True)
        try:
            detail = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
        except ValueError:
            detail = {"raw": r.stdout[-200:]}
        passed = (r.returncode == 0) and detail.get("correct", False)
        report["shapes"][s] = {"correct": passed, "detail": detail}
        if not passed:
            report["ok"] = False
    print(json.dumps(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
EOF

cat > .gitignore <<'EOF'
bin/
.warpspeed/
EOF

cat > README.md <<'EOF'
Mock kernel project generated by WarpSpeed/tests/mock/mkmockproj.sh.
build_command: ./build.sh    binary_path: bin/kernel_bench
kernel_paths: src/kernel.cu  harness: harness/ (read-only to agents)
EOF

chmod +x build.sh tools/render_bench.py harness/correctness.py

if [ ! -d .git ]; then
  git init -q
  git add -A
  git -c user.name=warpspeed-mock -c user.email=mock@warpspeed.local commit -qm "mock kernel project baseline (lat=$LAT profile=$PROFILE)"
fi

echo "mockproj ready at $(pwd)"
