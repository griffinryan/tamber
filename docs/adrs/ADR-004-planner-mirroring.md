# ADR-004 – Dual Planner Implementation (Rust + Python)

## Status
Accepted – Phase 0

## Context
Timbre requires composition planning to structure 90-180s music into coherent sections (Intro → Motif → Chorus → Outro). The planner determines:
- Tempo (68-128 BPM) based on duration and bar allocation
- Section templates (long-form vs. short-form)
- Bar distribution across sections
- Theme extraction from prompts
- Orchestration layer assignments
- Seed offsets for deterministic variation

The worker (Python) must execute planning for actual rendering. Question: Should the CLI (Rust) also have planning capability?

Alternatives considered:
1. **Python-only:** CLI requests plan from worker via HTTP
   - Pro: Single source of truth
   - Con: Network latency for previews
   - Con: Can't preview offline

2. **Rust-only:** CLI generates plans, sends to worker
   - Pro: Instant previews
   - Con: Worker must trust client plans (security/validation burden)
   - Con: Complex to enforce plan version compatibility

3. **Dual implementation with synchronization protocol (chosen):** Both Rust and Python implement identical planner logic
   - Pro: Instant offline previews
   - Pro: Worker remains authoritative (generates own plans)
   - Pro: Transparent to user (see what will be rendered)
   - Con: Maintenance burden (must keep synchronized)

## Decision
Implement planners in both Rust (`cli/src/planner.rs`) and Python (`worker/services/planner.py`) with strict synchronization protocol.

### Synchronization Requirements
1. **Constants must match exactly:**
   - Plan version (`PLAN_VERSION = "v3"`)
   - Tempo ranges (68-128 BPM)
   - Duration thresholds (90s for long-form)
   - Section priorities (ADD_PRIORITY, REMOVE_PRIORITY)
   - Keyword mappings (instruments, rhythms, textures)

2. **Template structures must match:**
   - Base bars, min/max bars
   - Section roles, energy levels
   - Prompt templates

3. **Algorithms must produce identical output:**
   - Tempo calculation
   - Bar allocation (including rebalancing logic)
   - Key selection
   - Orchestration assignment

4. **Testing strategy:**
   - Unit tests in both languages
   - Integration tests comparing outputs
   - Property-based tests for determinism
   - CI validation on every PR

### Change Protocol
When modifying planner logic:
1. Update Python first (source of truth)
2. Mirror changes in Rust
3. Update tests in both
4. Run sync validation script
5. Update documentation
6. Increment plan version if schema changes

## Consequences

### Benefits
✅ **Instant previews:** Users see plans immediately without network round-trip
✅ **Offline development:** Test CLI without worker running
✅ **Transparency:** Users understand what will be generated before submission
✅ **Validation:** Client can detect plan/backend mismatches early

### Costs
⚠️ **Maintenance burden:** Changes must be applied to both implementations
⚠️ **Synchronization risk:** Implementations may drift if not careful
⚠️ **Testing overhead:** Must validate both produce identical output
⚠️ **Documentation:** Requires synchronization protocol documentation

### Mitigation Strategies
- Comprehensive synchronization protocol (docs/PLANNER_SYNC.md)
- Automated diff testing in CI
- Golden file snapshot testing
- Clear ownership (Python is source of truth)
- Version guards (plan version increments on breaking changes)

### Trade-off Analysis
For Phase 0, benefits outweigh costs:
- User experience significantly improved (instant feedback)
- Offline capability valuable for development
- Maintenance burden manageable with good tooling
- Alternative (network-based preview) introduces latency and complexity

Future improvement: Code generation (generate Rust from Python) to reduce manual synchronization.

## Related Documents
- docs/PLANNER_SYNC.md – Full synchronization protocol
- docs/COMPOSITION.md – Planner v3 specification
- worker/services/planner.py – Python implementation
- cli/src/planner.rs – Rust implementation
