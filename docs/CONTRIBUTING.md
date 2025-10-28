# Contributing to Timbre

Thank you for your interest in contributing to Timbre! This document provides guidelines and workflows for contributors.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Organization](#code-organization)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Code Style](#code-style)
7. [Submitting Changes](#submitting-changes)
8. [Special Topics](#special-topics)

---

## Getting Started

### Prerequisites

- **macOS 13+** on Apple Silicon (primary development target)
- **Rust toolchain** (stable 1.75+)
- **Python 3.11** (strict, not 3.12+)
- **uv** package manager
- **Git** for version control

### First Steps

1. **Fork the repository**
   ```bash
   # On GitHub, click "Fork"
   git clone https://github.com/YOUR_USERNAME/tamber.git
   cd tamber
   ```

2. **Set up environment**
   ```bash
   make setup
   # This runs: cargo fetch + uv sync
   ```

3. **Verify setup**
   ```bash
   make test
   make lint
   ```

4. **Run the system**
   ```bash
   # Terminal 1: Start worker
   make worker-serve

   # Terminal 2: Run CLI
   make cli-run
   ```

---

## Development Setup

### Recommended Tools

- **Editor:** VS Code with Rust Analyzer + Python extensions
- **Terminal:** iTerm2, Alacritty, or WezTerm
- **Git UI:** GitKraken, Fork, or command line

### VS Code Extensions

```json
{
  "recommendations": [
    "rust-lang.rust-analyzer",
    "tamasfe.even-better-toml",
    "charliermarsh.ruff",
    "ms-python.python",
    "ms-python.vscode-pylance"
  ]
}
```

### Environment Variables

```bash
# Worker configuration
export TIMBRE_INFERENCE_DEVICE=mps  # or cpu, cuda
export TIMBRE_RIFFUSION_ALLOW_INFERENCE=1

# CLI configuration
export TIMBRE_WORKER_URL=http://localhost:8000
export TIMBRE_DEFAULT_DURATION=120
```

### Optional: Inference Dependencies

```bash
# For full audio generation (requires GPU or patience)
uv sync --project worker --extra inference

# For development without ML dependencies
uv sync --project worker --extra dev
# (Uses placeholder audio)
```

---

## Code Organization

### Repository Structure

```
tamber/
├── cli/                 # Rust TUI client
│   ├── src/
│   │   ├── main.rs      # Event loop, controller
│   │   ├── app.rs       # Application state
│   │   ├── ui/          # Ratatui rendering
│   │   ├── api.rs       # HTTP client
│   │   ├── planner.rs   # Planner mirror (keep synced!)
│   │   ├── types.rs     # Data models
│   │   └── config.rs    # Configuration loading
│   └── Cargo.toml
│
├── worker/              # Python FastAPI backend
│   ├── src/timbre_worker/
│   │   ├── app/         # FastAPI setup, routes, jobs
│   │   │   ├── main.py
│   │   │   ├── models.py
│   │   │   ├── routes.py
│   │   │   ├── jobs.py
│   │   │   └── settings.py
│   │   └── services/    # Core logic
│   │       ├── planner.py        # Planner (keep synced!)
│   │       ├── orchestrator.py   # Composition conductor
│   │       ├── musicgen.py       # MusicGen backend
│   │       ├── riffusion.py      # Riffusion backend
│   │       ├── audio_utils.py    # Mixing/mastering
│   │       └── types.py          # Shared types
│   ├── tests/
│   └── pyproject.toml
│
├── docs/                # Documentation
│   ├── MUSICGEN.md
│   ├── CONDITIONING.md
│   ├── PLANNER_SYNC.md  # READ THIS for planner changes!
│   └── adrs/            # Architecture decision records
│
├── scripts/             # Utility scripts
├── Makefile             # Development commands
└── CLAUDE.md            # Project instructions (for Claude Code)
```

### Key Files to Know

| File | Purpose | Update When |
|------|---------|-------------|
| `cli/src/planner.rs` | Rust planner | Planner logic changes |
| `worker/services/planner.py` | Python planner | Planner logic changes |
| `cli/src/types.rs` | Rust data models | API schema changes |
| `worker/app/models.py` | Python data models | API schema changes |
| `docs/PLANNER_SYNC.md` | Sync protocol | Planner changes |
| `CLAUDE.md` | AI assistant context | Project conventions change |

---

## Making Changes

### Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes**
   - Follow code style (see below)
   - Add tests
   - Update documentation

3. **Test locally**
   ```bash
   make test
   make lint
   ```

4. **Commit**
   ```bash
   git add .
   git commit -m "feat: add feature description"
   # or
   git commit -m "fix: resolve issue description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   # Then create PR on GitHub
   ```

### Commit Message Format

Use conventional commits:

```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic change)
- `refactor`: Code restructuring (no behavior change)
- `test`: Adding or updating tests
- `chore`: Maintenance (dependencies, tooling)

**Scope prefixes** (optional but recommended):
- `cli:` Rust client changes
- `worker:` Python backend changes
- `docs:` Documentation updates

**Examples:**
```
feat(cli): add /reset command to clear chat history
fix(worker): correct bar allocation rounding in planner
docs: add troubleshooting guide for MPS audio issues
chore(worker): update dependencies to latest versions
```

---

## Testing

### Running Tests

```bash
# All tests
make test

# Rust only
cargo test

# Python only
cd worker
uv run pytest

# Specific test
cargo test planner::test_long_form_plan
uv run pytest tests/test_planner.py::test_long_form_plan
```

### Writing Tests

**Rust:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        let result = my_function();
        assert_eq!(result, expected);
    }
}
```

**Python:**
```python
def test_feature():
    result = my_function()
    assert result == expected
```

### Test Coverage

- **Unit tests:** Core logic functions
- **Integration tests:** API endpoints, planner sync
- **Property tests:** Determinism, invariants

**Required for PRs:**
- New features must include tests
- Bug fixes should include regression tests
- Planner changes must pass sync validation

---

## Code Style

### Rust

**Formatting:**
```bash
cargo fmt
```

**Linting:**
```bash
cargo clippy -- -D warnings
```

**Style Guide:**
- Use `snake_case` for functions/variables
- Use `PascalCase` for types
- Prefer explicit types in public APIs
- Document public functions with `///`
- Max line length: 100 characters

**Example:**
```rust
/// Calculates tempo based on duration and bar allocation.
///
/// # Arguments
/// * `seconds_total` - Total composition duration
/// * `templates` - Section templates
/// * `beats_per_bar` - Beats per bar (usually 4)
///
/// # Returns
/// Tempo in BPM, clamped to 68-128
pub fn tempo_hint(seconds_total: f32, templates: &[SectionTemplate], beats_per_bar: u8) -> u16 {
    let total_weight: u32 = templates.iter().map(|t| t.base_bars as u32).sum();
    let beats = total_weight * beats_per_bar as u32;
    let raw = ((60.0 * beats as f32) / seconds_total.max(1.0)).round() as u16;
    select_tempo(raw)
}
```

### Python

**Formatting:**
```bash
cd worker
uv run ruff format src tests
```

**Linting:**
```bash
uv run ruff check src tests
uv run mypy
```

**Style Guide:**
- Follow PEP 8
- Use `snake_case` for functions/variables
- Use `PascalCase` for classes
- Type hints for all public functions
- Docstrings for public functions/classes
- Max line length: 100 characters

**Example:**
```python
def tempo_hint(
    total_seconds: float,
    templates: List[SectionTemplate],
    beats_per_bar: int,
) -> int:
    """Calculate tempo based on duration and bar allocation.

    Args:
        total_seconds: Total composition duration
        templates: Section templates
        beats_per_bar: Beats per bar (usually 4)

    Returns:
        Tempo in BPM, clamped to 68-128
    """
    total_weight = sum(tpl.base_bars for tpl in templates)
    beats = total_weight * beats_per_bar
    raw = int(round((60.0 * beats) / max(total_seconds, 1.0)))
    return max(MIN_TEMPO, min(raw, MAX_TEMPO))
```

---

## Submitting Changes

### Pull Request Process

1. **Ensure CI passes**
   - All tests pass
   - No linting errors
   - Documentation updated

2. **Write clear PR description**
   ```markdown
   ## Summary
   Brief description of changes

   ## Changes
   - Added feature X
   - Fixed bug Y
   - Updated documentation Z

   ## Testing
   - Ran all tests locally
   - Tested manually with [specific scenario]

   ## Checklist
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] Planner sync validated (if applicable)
   - [ ] CLAUDE.md updated (if conventions changed)
   ```

3. **Request review**
   - Tag relevant maintainers
   - Address review feedback promptly

4. **Squash commits** (if requested)
   ```bash
   git rebase -i main
   # Mark commits as "squash" or "fixup"
   ```

### PR Review Criteria

Reviewers will check:
- ✅ Code quality (follows style guide)
- ✅ Tests pass and provide good coverage
- ✅ Documentation updated
- ✅ Planner sync maintained (if applicable)
- ✅ No breaking changes (or properly documented)
- ✅ Commit messages clear and conventional

---

## Special Topics

### Modifying the Planner

**CRITICAL:** Planner exists in both Rust and Python. Changes must be synchronized.

**Process:**
1. Read `docs/PLANNER_SYNC.md` thoroughly
2. Update Python first (`worker/services/planner.py`)
3. Mirror changes in Rust (`cli/src/planner.rs`)
4. Update tests in both languages
5. Run sync validation (if script exists)
6. Update `docs/COMPOSITION.md`
7. Increment plan version if schema changes

**Checklist:**
- [ ] Constants match (tempo, thresholds, priorities)
- [ ] Templates match (bars, prompts, energy)
- [ ] Algorithms produce identical output
- [ ] Tests pass in both languages
- [ ] Documentation updated

### Adding Backend Parameters

When adding new generation parameters:

1. **Update Python models** (`worker/app/models.py`)
   ```python
   class GenerationRequest(BaseModel):
       new_param: Optional[float] = None
   ```

2. **Update Rust types** (`cli/src/types.rs`)
   ```rust
   pub struct GenerationRequest {
       pub new_param: Option<f32>,
   }
   ```

3. **Update backend service** (`worker/services/musicgen.py` or `riffusion.py`)
   ```python
   new_param_value = request.new_param or DEFAULT_NEW_PARAM
   ```

4. **Update extras metadata**
   ```python
   extras["new_param"] = new_param_value
   ```

5. **Update documentation** (`docs/MUSICGEN.md` or `docs/RIFFUSION.md`)

6. **Add CLI command** (if user-facing)
   ```rust
   // In app.rs
   Some("/newparam") => {
       self.generation_config.new_param = parse_value(parts);
   }
   ```

### Changing Mix Behavior

When modifying orchestrator mix logic:

1. **Update `orchestrator.py`** functions:
   - `_shape_to_target_length()`
   - `_butt_join()`
   - `_crossfade_seconds()`

2. **Update tests** (`worker/tests/test_orchestrator.py`)

3. **Update metadata** (track changes in extras)

4. **Update documentation** (`docs/AUDIO_PIPELINE.md`)

5. **Consider versioning** (if behavior changes significantly)

### Adding Dependencies

**Rust:**
```bash
# Add to Cargo.toml
cargo add package-name

# Or manually edit Cargo.toml
```

**Python:**
```bash
cd worker
# Edit pyproject.toml dependencies section
uv sync
```

**For inference dependencies (optional):**
```toml
[project.optional-dependencies]
inference = [
    "torch>=2.0",
    "transformers>=4.30",
    # ...
]
```

---

## Community

### Communication

- **Issues:** Use GitHub Issues for bugs and feature requests
- **Discussions:** Use GitHub Discussions for questions and ideas
- **PRs:** Use pull requests for code contributions

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help newcomers learn

### Recognition

Contributors will be:
- Listed in commit history
- Acknowledged in release notes
- Credited in documentation (for major contributions)

---

## Resources

### Documentation
- `docs/architecture.md` - System overview
- `docs/setup.md` - Detailed setup guide
- `docs/MUSICGEN.md` - MusicGen backend
- `docs/CONDITIONING.md` - Audio conditioning
- `docs/PLANNER_SYNC.md` - Planner synchronization
- `docs/TROUBLESHOOTING.md` - Common issues

### External Resources
- [Rust Book](https://doc.rust-lang.org/book/)
- [Ratatui Documentation](https://docs.rs/ratatui/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MusicGen Paper](https://arxiv.org/abs/2306.05284)

---

## Questions?

If you have questions not covered here:
1. Check `docs/TROUBLESHOOTING.md`
2. Search existing GitHub Issues
3. Open a new Discussion
4. Reach out to maintainers

---

**Thank you for contributing to Timbre!**

Your contributions help make text-to-music generation accessible and powerful.

---

**Document Version:** 1.0
**Last Updated:** 2025
**License:** [Your License Here]
