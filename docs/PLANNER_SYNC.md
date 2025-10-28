# Planner Synchronization Protocol

This document defines the protocol for maintaining synchronization between the Rust and Python planner implementations, ensuring deterministic plan generation across both environments.

---

## 1. Why Two Planners?

### Design Rationale

**Python Planner** (`worker/services/planner.py`):
- **Purpose:** Server-side plan generation for actual audio rendering
- **Context:** Has access to backend status, can make runtime decisions
- **Usage:** Every generation request uses this planner

**Rust Planner** (`cli/src/planner.rs`):
- **Purpose:** Client-side plan preview before submission
- **Context:** Offline, no backend access
- **Usage:** CLI displays plans for user review

### Benefits of Mirroring

1. **Instant Preview:** Users see plans without network round-trip
2. **Offline Development:** Test plan logic without worker running
3. **Transparency:** Users understand what will be generated
4. **Validation:** Client can detect plan/backend mismatches early

### Cost of Mirroring

- **Maintenance burden:** Changes must be applied to both implementations
- **Synchronization risk:** Implementations may drift if not careful
- **Testing overhead:** Must validate both produce identical output

**Trade-off Decision:** Benefits outweigh costs for Phase 0 (validated by ADR-004)

---

## 2. Synchronization Invariants

### CRITICAL: These MUST Match Exactly

#### Plan Version
```rust
// Rust: cli/src/planner.rs
const PLAN_VERSION: &str = "v3";
```
```python
# Python: worker/services/planner.py
PLAN_VERSION = "v3"
```

**When to update:** Increment when plan structure changes (add/remove fields, change semantics)

#### Tempo Constants
```rust
// Rust
const MIN_TEMPO: u16 = 68;
const MAX_TEMPO: u16 = 128;
```
```python
# Python
MIN_TEMPO = 68
MAX_TEMPO = 128
```

#### Duration Thresholds
```rust
// Rust
const LONG_FORM_THRESHOLD: f32 = 90.0;
const LONG_MIN_SECTION_SECONDS: f32 = 16.0;
const SHORT_MIN_SECTION_SECONDS: f32 = 2.0;
const SHORT_MIN_TOTAL_SECONDS: f32 = 2.0;
```
```python
# Python
LONG_FORM_THRESHOLD = 90.0
LONG_MIN_SECTION_SECONDS = 16.0
SHORT_MIN_SECTION_SECONDS = 2.0
SHORT_MIN_TOTAL_SECONDS = 2.0
```

#### Section Role Priorities
```rust
// Rust: Bar allocation priorities
const ADD_PRIORITY: &[SectionRole] = &[
    SectionRole::Chorus,
    SectionRole::Motif,
    SectionRole::Bridge,
    SectionRole::Development,
    SectionRole::Intro,
    SectionRole::Outro,
    SectionRole::Resolution,
];

const REMOVE_PRIORITY: &[SectionRole] = &[
    SectionRole::Outro,
    SectionRole::Intro,
    SectionRole::Bridge,
    SectionRole::Resolution,
    SectionRole::Development,
    SectionRole::Motif,
    SectionRole::Chorus,
];
```
```python
# Python
ADD_PRIORITY = [
    SectionRole.CHORUS,
    SectionRole.MOTIF,
    SectionRole.BRIDGE,
    SectionRole.DEVELOPMENT,
    SectionRole.INTRO,
    SectionRole.OUTRO,
    SectionRole.RESOLUTION,
]

REMOVE_PRIORITY = [
    SectionRole.OUTRO,
    SectionRole.INTRO,
    SectionRole.BRIDGE,
    SectionRole.RESOLUTION,
    SectionRole.DEVELOPMENT,
    SectionRole.MOTIF,
    SectionRole.CHORUS,
]
```

**Critical:** Order matters! Used for bar allocation rebalancing.

#### Keyword Mappings
```rust
// Rust: Exact same mappings
const INSTRUMENT_KEYWORDS: &[(&str, &str)] = &[
    ("piano", "warm piano"),
    ("keys", "soft keys"),
    ("synthwave", "retro synth layers"),
    // ... (36+ mappings)
];

const RHYTHM_KEYWORDS: &[(&str, &str)] = &[
    ("waltz", "gentle 3/4 sway"),
    // ... (12+ mappings)
];

const TEXTURE_KEYWORDS: &[(&str, &str)] = &[
    ("dream", "dreamy haze"),
    // ... (10+ mappings)
];
```
```python
# Python: Identical dictionaries
INSTRUMENT_KEYWORDS = {
    "piano": "warm piano",
    "keys": "soft keys",
    "synthwave": "retro synth layers",
    # ... (exact same 36+ mappings)
}

RHYTHM_KEYWORDS = {
    "waltz": "gentle 3/4 sway",
    # ... (exact same 12+ mappings)
}

TEXTURE_KEYWORDS = {
    "dream": "dreamy haze",
    # ... (exact same 10+ mappings)
}
```

**Testing:** Use integration tests to verify keyword matching produces identical results.

---

## 3. Template Structures

### Long-Form Templates

Both implementations must define identical templates for each duration range.

**Duration ≥150s:**
```rust
// Rust
templates = vec![
    SectionTemplate { role: Intro, base_bars: 16, ... },
    SectionTemplate { role: Motif, base_bars: 20, ... },
    SectionTemplate { role: Bridge, base_bars: 12, ... },
    SectionTemplate { role: Chorus, base_bars: 24, ... },
    SectionTemplate { role: Outro, base_bars: 14, ... },
]
```
```python
# Python
templates = [
    SectionTemplate(role=SectionRole.INTRO, base_bars=16, ...),
    SectionTemplate(role=SectionRole.MOTIF, base_bars=20, ...),
    SectionTemplate(role=SectionRole.BRIDGE, base_bars=12, ...),
    SectionTemplate(role=SectionRole.CHORUS, base_bars=24, ...),
    SectionTemplate(role=SectionRole.OUTRO, base_bars=14, ...),
]
```

**Duration 90-150s:**
```rust
// Rust: No bridge
templates = vec![
    SectionTemplate { role: Intro, base_bars: 12, ... },
    SectionTemplate { role: Motif, base_bars: 18, ... },
    SectionTemplate { role: Chorus, base_bars: 24, ... },
    SectionTemplate { role: Outro, base_bars: 12, ... },
]
```
```python
# Python: Exact same
templates = [
    SectionTemplate(role=SectionRole.INTRO, base_bars=12, ...),
    SectionTemplate(role=SectionRole.MOTIF, base_bars=18, ...),
    SectionTemplate(role=SectionRole.CHORUS, base_bars=24, ...),
    SectionTemplate(role=SectionRole.OUTRO, base_bars=12, ...),
]
```

### Short-Form Templates

**Duration ≥24s:**
```rust
// Rust
templates = vec![
    SectionTemplate { role: Intro, base_bars: 4, ... },
    SectionTemplate { role: Motif, base_bars: 8, ... },
    SectionTemplate { role: Development, base_bars: 8, ... },
    SectionTemplate { role: Resolution, base_bars: 4, ... },
    SectionTemplate { role: Outro, base_bars: 4, ... },
]
```
```python
# Python
templates = [
    SectionTemplate(role=SectionRole.INTRO, base_bars=4, ...),
    SectionTemplate(role=SectionRole.MOTIF, base_bars=8, ...),
    SectionTemplate(role=SectionRole.DEVELOPMENT, base_bars=8, ...),
    SectionTemplate(role=SectionRole.RESOLUTION, base_bars=4, ...),
    SectionTemplate(role=SectionRole.OUTRO, base_bars=4, ...),
]
```

**Duration 16-24s, <16s:** (Similar pattern, must match exactly)

### Template Fields
```rust
// Rust
struct SectionTemplate {
    role: SectionRole,
    label: &'static str,
    energy: SectionEnergy,
    base_bars: u16,
    min_bars: u16,
    max_bars: u16,
    prompt_template: &'static str,
    transition: Option<&'static str>,
}
```
```python
# Python
@dataclass
class SectionTemplate:
    role: SectionRole
    label: str
    energy: SectionEnergy
    base_bars: int
    min_bars: int
    max_bars: int
    prompt_template: str
    transition: Optional[str]
```

**All fields must match:** Different `base_bars` will cause divergent bar allocations.

---

## 4. Algorithm Parity

### Tempo Calculation

**Python:**
```python
def _tempo_hint(total_seconds: float, templates: List, beats_per_bar: int) -> int:
    total_weight = sum(tpl.base_bars for tpl in templates)
    beats = total_weight * beats_per_bar
    raw = int(round((60.0 * beats) / max(total_seconds, 1.0)))
    return max(MIN_TEMPO, min(raw, MAX_TEMPO))
```

**Rust:**
```rust
fn tempo_hint(seconds_total: f32, templates: &[SectionTemplate], beats_per_bar: u8) -> u16 {
    let total_weight: u32 = templates.iter().map(|t| t.base_bars as u32).sum();
    let beats = total_weight * beats_per_bar as u32;
    let raw = ((60.0 * beats as f32) / seconds_total.max(1.0)).round() as u16;
    select_tempo(raw)  // Clamps to MIN_TEMPO..MAX_TEMPO
}
```

**Must match:** Same inputs → same tempo

### Bar Allocation

**Python:**
```python
def _allocate_bars(templates, total_seconds, seconds_per_bar):
    # 1. Initial scaled allocation
    target_bars = int(round(total_seconds / seconds_per_bar))
    allocated = [
        int(round(tpl.base_bars * (target_bars / total_weight)))
        for tpl in templates
    ]

    # 2. Rebalance using priorities
    while sum(allocated) < target_bars:
        for role in ADD_PRIORITY:
            idx = _find_role_index(templates, role)
            if allocated[idx] < templates[idx].max_bars:
                allocated[idx] += 1
                break

    while sum(allocated) > target_bars:
        for role in REMOVE_PRIORITY:
            idx = _find_role_index(templates, role)
            if allocated[idx] > templates[idx].min_bars:
                allocated[idx] -= 1
                break

    return allocated
```

**Rust:**
```rust
fn allocate_bars(templates: &[SectionTemplate], seconds_total: f32, seconds_per_bar: f32) -> Vec<u16> {
    // 1. Initial scaled allocation
    let target_bars = (seconds_total / seconds_per_bar).round() as u16;
    let total_weight: u32 = templates.iter().map(|t| t.base_bars as u32).sum();
    let mut allocated: Vec<u16> = templates
        .iter()
        .map(|tpl| {
            ((tpl.base_bars as f32 * target_bars as f32) / total_weight as f32).round() as u16
        })
        .collect();

    // 2. Rebalance using priorities
    while allocated.iter().sum::<u16>() < target_bars {
        for &role in ADD_PRIORITY {
            if let Some(idx) = templates.iter().position(|t| t.role == role) {
                if allocated[idx] < templates[idx].max_bars {
                    allocated[idx] += 1;
                    break;
                }
            }
        }
    }

    while allocated.iter().sum::<u16>() > target_bars {
        for &role in REMOVE_PRIORITY {
            if let Some(idx) = templates.iter().position(|t| t.role == role) {
                if allocated[idx] > templates[idx].min_bars {
                    allocated[idx] -= 1;
                    break;
                }
            }
        }
    }

    allocated
}
```

**Critical:** Rounding strategy, iteration order, priority lists must all match.

### Key Selection

**Python:**
```python
def _select_key(seed: int) -> str:
    keys = [
        "C major", "G major", "D major", "A major", "E major", "B major",
        "F major", "Bb major", "Eb major", "Ab major", "Db major", "Gb major",
        "A minor", "E minor", "B minor", "F# minor", "C# minor", "G# minor",
        "D minor", "G minor", "C minor", "F minor", "Bb minor", "Eb minor",
    ]
    return keys[seed % len(keys)]
```

**Rust:**
```rust
fn select_key(seed: u64) -> String {
    let keys = [
        "C major", "G major", "D major", "A major", "E major", "B major",
        "F major", "Bb major", "Eb major", "Ab major", "Db major", "Gb major",
        "A minor", "E minor", "B minor", "F# minor", "C# minor", "G# minor",
        "D minor", "G minor", "C minor", "F minor", "Bb minor", "Eb minor",
    ];
    keys[(seed as usize) % keys.len()].to_string()
}
```

**Must match:** Key list order, modulo operation

---

## 5. Orchestration Layer Assignment

### Layer Profiles

**Python:**
```python
SECTION_LAYER_PROFILE = {
    SectionRole.INTRO: LayerCounts(rhythm=1, bass=1, harmony=3, lead=1, textures=2, vocals=0),
    SectionRole.MOTIF: LayerCounts(rhythm=2, bass=1, harmony=2, lead=2, textures=1, vocals=0),
    SectionRole.CHORUS: LayerCounts(rhythm=2, bass=1, harmony=2, lead=2, textures=1, vocals=1),
    SectionRole.BRIDGE: LayerCounts(rhythm=1, bass=1, harmony=3, lead=1, textures=2, vocals=0),
    SectionRole.DEVELOPMENT: LayerCounts(rhythm=2, bass=1, harmony=2, lead=1, textures=1, vocals=0),
    SectionRole.RESOLUTION: LayerCounts(rhythm=1, bass=1, harmony=2, lead=1, textures=1, vocals=0),
    SectionRole.OUTRO: LayerCounts(rhythm=1, bass=1, harmony=2, lead=1, textures=2, vocals=0),
}
```

**Rust:**
```rust
fn layer_profile(role: &SectionRole) -> LayerCounts {
    match role {
        SectionRole::Intro => LayerCounts::new(1, 1, 3, 1, 2, 0),
        SectionRole::Motif => LayerCounts::new(2, 1, 2, 2, 1, 0),
        SectionRole::Chorus => LayerCounts::new(2, 1, 2, 2, 1, 1),
        SectionRole::Bridge => LayerCounts::new(1, 1, 3, 1, 2, 0),
        SectionRole::Development => LayerCounts::new(2, 1, 2, 1, 1, 0),
        SectionRole::Resolution => LayerCounts::new(1, 1, 2, 1, 1, 0),
        SectionRole::Outro => LayerCounts::new(1, 1, 2, 1, 2, 0),
    }
}
```

### Category Assignment

**Python:**
```python
CATEGORY_KEYWORDS = {
    "drums": "rhythm",
    "percussion": "rhythm",
    "bass": "bass",
    "piano": "harmony",
    "keys": "harmony",
    "guitar": "lead",
    "synth": "harmony",
    "strings": "harmony",
    "vocals": "vocals",
    "choir": "vocals",
    # ... etc
}
```

**Rust:**
```rust
fn categorize_instrument(instrument: &str) -> &'static str {
    match instrument.to_lowercase().as_str() {
        s if s.contains("drum") => "rhythm",
        s if s.contains("percussion") => "rhythm",
        s if s.contains("bass") => "bass",
        s if s.contains("piano") || s.contains("keys") => "harmony",
        s if s.contains("guitar") => "lead",
        // ... etc (exact same logic)
    }
}
```

**Must match:** Categorization logic, fallback defaults

---

## 6. Directives & Metadata

### Motif Directives

**Python:**
```python
def _directives_for_role(role: SectionRole):
    directives = {
        SectionRole.INTRO: (
            "foreshadow motif",
            ["texture", "register preview"],
            "establish tonic pedal"
        ),
        SectionRole.MOTIF: (
            "state motif",
            ["motif fidelity"],
            "open cadence"
        ),
        # ... etc
    }
    return directives[role]
```

**Rust:**
```rust
fn directives_for_role(role: &SectionRole) -> (Option<String>, Vec<String>, Option<String>) {
    match role {
        SectionRole::Intro => (
            Some("foreshadow motif".to_string()),
            vec!["texture".to_string(), "register preview".to_string()],
            Some("establish tonic pedal".to_string()),
        ),
        SectionRole::Motif => (
            Some("state motif".to_string()),
            vec!["motif fidelity".to_string()],
            Some("open cadence".to_string()),
        ),
        // ... etc (exact same)
    }
}
```

**Must match:** Directive strings, variation axes, cadence hints

---

## 7. Testing Strategy

### Unit Tests

**Python:**
```python
# worker/tests/test_planner.py
def test_long_form_plan_structure():
    planner = CompositionPlanner()
    plan = planner.build_plan("dreamy piano", duration_seconds=120, seed=42)

    assert plan.version == "v3"
    assert plan.tempo_bpm >= 68 and plan.tempo_bpm <= 128
    assert len(plan.sections) == 4  # Intro, Motif, Chorus, Outro
    assert plan.sections[0].role == SectionRole.INTRO
    assert plan.sections[1].role == SectionRole.MOTIF
    # ... etc
```

**Rust:**
```rust
// cli/src/planner.rs (module tests at bottom)
#[test]
fn test_long_form_plan_structure() {
    let planner = CompositionPlanner::new();
    let plan = planner.build_plan("dreamy piano", 120, Some(42));

    assert_eq!(plan.version, "v3");
    assert!(plan.tempo_bpm >= 68 && plan.tempo_bpm <= 128);
    assert_eq!(plan.sections.len(), 4);
    assert_eq!(plan.sections[0].role, SectionRole::Intro);
    assert_eq!(plan.sections[1].role, SectionRole::Motif);
    // ... etc
}
```

### Integration Tests

**Cross-Language Validation:**
```bash
# Test script: scripts/test_planner_sync.sh
#!/bin/bash

# Generate plan from Python
python -c "
from worker.services.planner import CompositionPlanner
import json
planner = CompositionPlanner()
plan = planner.build_plan('dreamy piano', 120, seed=42)
print(json.dumps(plan.model_dump(), indent=2))
" > plan_python.json

# Generate plan from Rust
cargo run --bin test-planner -- "dreamy piano" 120 42 > plan_rust.json

# Compare (excluding timestamps, UUIDs, etc.)
diff <(jq 'del(.timestamp)' plan_python.json) \
     <(jq 'del(.timestamp)' plan_rust.json)

# Exit 0 if identical
```

**CI Integration:**
```yaml
# .github/workflows/test-planner-sync.yml
name: Planner Sync Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: dtolnay/rust-toolchain@stable
      - uses: astral-sh/setup-uv@v1
      - run: ./scripts/test_planner_sync.sh
```

### Property-Based Testing

**Python (hypothesis):**
```python
from hypothesis import given, strategies as st

@given(
    prompt=st.text(min_size=1, max_size=100),
    duration=st.integers(min_value=90, max_value=180),
    seed=st.integers(min_value=0, max_value=2**32-1)
)
def test_plan_determinism(prompt, duration, seed):
    planner = CompositionPlanner()
    plan1 = planner.build_plan(prompt, duration, seed)
    plan2 = planner.build_plan(prompt, duration, seed)
    assert plan1 == plan2  # Same inputs → same output
```

**Rust (proptest):**
```rust
#[cfg(test)]
mod proptests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn plan_determinism(
            prompt in ".+",
            duration in 90u8..=180u8,
            seed in any::<u64>()
        ) {
            let planner = CompositionPlanner::new();
            let plan1 = planner.build_plan(&prompt, duration, Some(seed));
            let plan2 = planner.build_plan(&prompt, duration, Some(seed));
            assert_eq!(plan1, plan2);
        }
    }
}
```

---

## 8. Change Protocol

### When Modifying Planner Logic

**Required Steps:**
1. ✅ **Update Python first** (`worker/services/planner.py`)
2. ✅ **Mirror in Rust** (`cli/src/planner.rs`)
3. ✅ **Update tests** (both Python and Rust)
4. ✅ **Run sync validation** (`./scripts/test_planner_sync.sh`)
5. ✅ **Update docs** (`docs/COMPOSITION.md`, this file)
6. ✅ **Increment plan version** (if schema changes)

### Checklist for Constants
When adding/modifying constants:
- [ ] Python constant added/updated
- [ ] Rust constant added/updated
- [ ] Values match exactly (including order for arrays)
- [ ] Types compatible (f32 ↔ float, u16 ↔ int, etc.)
- [ ] Tests updated to validate new constant

### Checklist for Templates
When modifying templates:
- [ ] Template structure matches (fields, order)
- [ ] Base bars, min/max bars match
- [ ] Prompt templates match (exact strings)
- [ ] Energy levels match
- [ ] Transition hints match
- [ ] Duration thresholds updated if needed

### Checklist for Algorithms
When modifying algorithms:
- [ ] Python algorithm updated
- [ ] Rust algorithm mirrors logic (not just behavior)
- [ ] Rounding strategy matches
- [ ] Iteration order matches (critical for bar allocation)
- [ ] Edge cases handled identically
- [ ] Property tests pass

---

## 9. Common Pitfalls

### Pitfall 1: Floating-Point Precision
**Problem:**
```python
# Python
tempo = int(round(120.5))  # → 121

// Rust
let tempo = 120.5f32.round() as u16;  // → 120 (different rounding!)
```

**Solution:** Use consistent rounding strategies
```rust
// Rust: Match Python's "round half to even"
let tempo = (120.5 + 0.5).floor() as u16;  // → 121
```

### Pitfall 2: Integer Division
**Problem:**
```python
# Python 3
bars = 10 / 3  # → 3.333... (float division)

// Rust
let bars = 10 / 3;  // → 3 (integer division)
```

**Solution:** Explicit float division
```rust
let bars = 10.0 / 3.0;  // → 3.333...
```

### Pitfall 3: String Matching
**Problem:**
```python
# Python: case-insensitive
if "Piano" in prompt.lower():

// Rust: case-sensitive
if prompt.contains("Piano") {  // Misses "piano"!
```

**Solution:** Normalize case
```rust
if prompt.to_lowercase().contains("piano") {
```

### Pitfall 4: Array Indexing
**Problem:**
```python
# Python: negative indexing
keys = [...]
key = keys[-1]  # Last element

// Rust: no negative indexing
let key = keys[-1];  // COMPILE ERROR
```

**Solution:**
```rust
let key = keys[keys.len() - 1];  // Last element
```

### Pitfall 5: Modulo with Negative Numbers
**Problem:**
```python
# Python
-5 % 3  # → 1

// Rust
-5 % 3  // → -2 (different!)
```

**Solution:** Use positive inputs or rem_euclid
```rust
(-5i32).rem_euclid(3)  // → 1 (matches Python)
```

---

## 10. Version Management

### Incrementing Plan Version

**When to increment:**
- Schema changes (add/remove fields)
- Semantic changes (same fields, different meaning)
- Algorithm changes that affect output structure

**When NOT to increment:**
- Bug fixes that restore intended behavior
- Refactoring without behavioral changes
- Changes to internal helper functions

**Process:**
1. Increment version string in both files
   ```rust
   const PLAN_VERSION: &str = "v4";  // was v3
   ```
   ```python
   PLAN_VERSION = "v4"
   ```

2. Update version guards in consumers
   ```python
   # orchestrator.py
   def generate(...):
       if plan.version != "v4":
           raise ValueError(f"Unsupported plan version: {plan.version}")
   ```

3. Document migration in `docs/COMPOSITION.md`

4. Add migration tests
   ```python
   def test_v3_to_v4_migration():
       old_plan = load_v3_plan()
       new_plan = migrate_plan(old_plan)
       assert new_plan.version == "v4"
   ```

---

## 11. Troubleshooting

### Issue: Plans Diverge Between Rust and Python

**Symptoms:**
- Different tempo for same inputs
- Different bar allocations
- Different key selection

**Diagnosis:**
```bash
# Generate plans with identical inputs
python -c "..." > plan_py.json
cargo run --bin test-planner > plan_rust.json

# Compare field by field
diff plan_py.json plan_rust.json

# Look for:
- Different tempo_bpm
- Different total_bars
- Different section.bars values
- Different section.orchestration
```

**Common Causes:**
1. **Constants out of sync:** Check all CRITICAL constants match
2. **Template mismatch:** Verify template structures identical
3. **Algorithm divergence:** Review bar allocation, tempo calculation
4. **Floating-point rounding:** Check rounding strategies match

**Fix:**
1. Identify divergent field
2. Trace backwards to source constant/algorithm
3. Update one implementation to match the other (usually update Rust to match Python)
4. Re-test

### Issue: Tests Pass but Real Outputs Differ

**Possible Causes:**
- Tests using simplified inputs
- Edge cases not covered
- Floating-point accumulation errors

**Solution:**
Add real-world test cases:
```python
# Test with actual user prompts
def test_real_world_prompt_1():
    plan = planner.build_plan(
        "nostalgic synthwave with dreamy lo-fi piano and atmospheric textures",
        duration_seconds=150,
        seed=12345
    )
    # Snapshot test or golden file comparison
    assert plan.matches_golden("test_data/golden_plan_1.json")
```

---

## 12. Future Improvements

### Potential Enhancements

1. **Code Generation:**
   - Generate Rust from Python (or vice versa)
   - Reduces manual synchronization burden
   - Requires sophisticated code generation tooling

2. **Shared Schema:**
   - Define plan structure in JSON Schema or Protobuf
   - Generate both Python and Rust types
   - Single source of truth

3. **Runtime Validation:**
   - CLI validates plans against schema before rendering
   - Detects drift automatically
   - Fails fast on mismatch

4. **Formal Verification:**
   - Prove equivalence using theorem provers
   - High confidence in synchronization
   - Requires significant tooling investment

### Interim Improvements

1. **Automated Diff Testing:**
   ```bash
   # CI runs on every PR
   ./scripts/diff_planners.sh --seed-range 0:1000
   ```

2. **Snapshot Testing:**
   ```python
   # Capture plan outputs, detect changes
   from syrupy import SnapshotTest
   def test_plan_snapshot(snapshot):
       plan = planner.build_plan("test", 120, 42)
       assert plan.model_dump() == snapshot
   ```

3. **Continuous Monitoring:**
   - Track planner drift metrics in CI
   - Alert on divergence
   - Dashboard showing sync health

---

## 13. References

- **ADR-004:** Planner Mirroring Decision (docs/adrs/ADR-004-planner-mirroring.md)
- **Planner v3 Spec:** docs/COMPOSITION.md
- **Python Implementation:** worker/services/planner.py
- **Rust Implementation:** cli/src/planner.rs
- **Test Suite:** worker/tests/test_planner.py, cli/src/planner.rs (module tests)

---

**Document Version:** 1.0
**Last Updated:** 2025
**Current Plan Version:** v3
**Sync Status:** ✅ Synchronized (validated 2025)
