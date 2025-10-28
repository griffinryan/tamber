# Rust TUI Implementation Guide

This document provides a comprehensive technical guide to Timbre's Rust-based terminal UI built with Ratatui.

---

## 1. Architecture Overview

### Component Hierarchy

```
main.rs
  ├─ Controller (background async tasks)
  │   ├─ HTTP Client (reqwest)
  │   ├─ Poll Tasks (job status monitoring)
  │   └─ AudioPlayer (rodio)
  │
  ├─ AppState (UI state management)
  │   ├─ Chat History
  │   ├─ Job Registry
  │   ├─ Generation Config
  │   └─ Planner Mirror
  │
  └─ UI Renderer (Ratatui)
      ├─ Status Bar
      ├─ Conversation Pane
      ├─ Prompt Input
      ├─ Jobs Pane
      ├─ Backends Pane
      └─ Status Pane
```

### Event Flow

```
User Input (keyboard/mouse)
    ↓
Event Handler (main.rs)
    ↓
AppState mutation (app.rs)
    ↓
UI Re-render (ui/mod.rs)
    ↓
Terminal Display
```

**Async Events:**
```
Background Task (Controller)
    ↓
app_event_tx.send(AppEvent::JobUpdated)
    ↓
Event Loop receives
    ↓
AppState mutation
    ↓
UI Re-render
```

---

## 2. Main Event Loop (`main.rs`)

### Initialization

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // 1. Load configuration
    let config = Config::load()?;

    // 2. Create HTTP client
    let client = Client::new(&config.worker_url);

    // 3. Health check
    let health = client.health().await?;

    // 4. Setup terminal
    enable_raw_mode()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;

    // 5. Create app state
    let mut app = AppState::new(config, health);

    // 6. Create event channels
    let (app_event_tx, mut app_event_rx) = mpsc::unbounded_channel::<AppEvent>();
    let (command_tx, command_rx) = mpsc::unbounded_channel::<AppCommand>();

    // 7. Spawn controller
    Controller::spawn(client.clone(), command_rx, app_event_tx.clone());

    // 8. Event loop
    loop {
        // Render UI
        terminal.draw(|f| ui::render(f, &mut app))?;

        // Handle events (keyboard, app events)
        if !handle_events(&mut app, &command_tx, &mut app_event_rx).await? {
            break;  // User quit
        }
    }

    // 9. Cleanup
    disable_raw_mode()?;
    terminal.show_cursor()?;

    Ok(())
}
```

### Event Handling

```rust
async fn handle_events(
    app: &mut AppState,
    command_tx: &UnboundedSender<AppCommand>,
    app_event_rx: &mut UnboundedReceiver<AppEvent>,
) -> Result<bool> {
    // Poll for events with timeout
    let timeout = Duration::from_millis(100);

    tokio::select! {
        // Keyboard input
        Ok(event) = tokio::time::timeout(timeout, read_event()) => {
            match event {
                Event::Key(key) => {
                    if !handle_key_event(key, app, command_tx)? {
                        return Ok(false);  // Quit
                    }
                }
                _ => {}
            }
        }

        // App events from background tasks
        Some(app_event) = app_event_rx.recv() => {
            handle_app_event(app_event, app);
        }

        // Timeout (re-render)
        _ = tokio::time::sleep(timeout) => {}
    }

    Ok(true)
}
```

---

## 3. Application State (`app.rs`)

### AppState Structure

```rust
pub struct AppState {
    // User input
    pub input: String,
    pub input_mode: InputMode,  // Normal or Insert

    // Chat history
    pub chat: Vec<ChatEntry>,
    pub chat_scroll: usize,
    pub chat_following: bool,  // Auto-scroll to bottom

    // Job management
    pub jobs: IndexMap<String, JobEntry>,  // Insertion-order map
    pub selected_job_index: Option<usize>,
    pub jobs_scroll: usize,

    // Status tracking
    pub status_lines: Vec<String>,  // Max 8 lines
    pub last_health: Option<HealthResponse>,

    // Playback
    pub playback: Option<PlaybackState>,

    // Configuration
    pub generation_config: GenerationConfig,
    pub artifact_dir: PathBuf,

    // Planner mirror
    pub planner: CompositionPlanner,

    // UI state
    pub active_pane: Pane,  // Which pane has focus
}
```

### Key Data Structures

```rust
pub struct ChatEntry {
    pub timestamp: DateTime<Local>,
    pub role: ChatRole,  // User, Worker, System
    pub content: String,
}

pub struct JobEntry {
    pub job_id: String,
    pub status: GenerationStatus,
    pub request: GenerationRequest,
    pub artifact: Option<GenerationArtifact>,
    pub created_at: DateTime<Local>,
}

pub struct PlaybackState {
    pub job_id: String,
    pub duration_seconds: Option<f32>,
    pub started_at: Instant,
}

pub struct GenerationConfig {
    pub model_id: String,
    pub duration_seconds: u8,
    pub cfg_scale: Option<f32>,
    pub seed: Option<u64>,
}
```

### Command Handling

```rust
impl AppState {
    pub fn handle_command(&mut self, input: &str) -> Result<Option<AppCommand>> {
        let parts: Vec<&str> = input.split_whitespace().collect();

        match parts.get(0).map(|s| *s) {
            Some("/duration") => {
                let duration = parts.get(1)
                    .and_then(|s| s.parse::<u8>().ok())
                    .unwrap_or(120);
                self.generation_config.duration_seconds = duration.clamp(90, 180);
                Ok(None)
            }

            Some("/model") => {
                if let Some(model) = parts.get(1) {
                    self.generation_config.model_id = model.to_string();
                }
                Ok(None)
            }

            Some("/cfg") => {
                match parts.get(1).map(|s| *s) {
                    Some("off") => self.generation_config.cfg_scale = None,
                    Some(value) => {
                        self.generation_config.cfg_scale = value.parse().ok();
                    }
                    None => {}
                }
                Ok(None)
            }

            Some("/seed") => {
                match parts.get(1).map(|s| *s) {
                    Some("off") => self.generation_config.seed = None,
                    Some(value) => {
                        self.generation_config.seed = value.parse().ok();
                    }
                    None => {}
                }
                Ok(None)
            }

            _ => {
                // Not a command, submit as prompt
                Ok(Some(AppCommand::SubmitPrompt {
                    prompt: input.to_string(),
                }))
            }
        }
    }
}
```

---

## 4. Controller & Background Tasks (`main.rs`)

### Controller Spawn

```rust
pub struct Controller;

impl Controller {
    pub fn spawn(
        client: Arc<Client>,
        mut command_rx: UnboundedReceiver<AppCommand>,
        app_event_tx: UnboundedSender<AppEvent>,
    ) {
        tokio::spawn(async move {
            while let Some(command) = command_rx.recv().await {
                match command {
                    AppCommand::SubmitPrompt { prompt } => {
                        Self::handle_submit_prompt(
                            &client,
                            prompt,
                            &app_event_tx,
                        ).await;
                    }

                    AppCommand::PlayJob { job_id } => {
                        Self::handle_play_job(
                            &client,
                            job_id,
                            &app_event_tx,
                        ).await;
                    }

                    AppCommand::StopPlayback => {
                        // Stop audio playback
                    }
                }
            }
        });
    }
}
```

### Submit Prompt Flow

```rust
async fn handle_submit_prompt(
    client: &Client,
    prompt: String,
    app_event_tx: &UnboundedSender<AppEvent>,
) {
    // 1. Build request
    let request = GenerationRequest {
        prompt: prompt.clone(),
        duration_seconds: config.duration_seconds,
        model_id: config.model_id.clone(),
        seed: config.seed,
        cfg_scale: config.cfg_scale,
        // ... other fields
    };

    // 2. Submit to worker
    match client.submit_generation(request).await {
        Ok(status) => {
            // 3. Send JobQueued event
            app_event_tx.send(AppEvent::JobQueued {
                job_id: status.job_id.clone(),
                status,
                request,
            }).ok();

            // 4. Spawn poll task
            spawn_poll_task(
                client.clone(),
                status.job_id,
                app_event_tx.clone(),
            );
        }

        Err(err) => {
            app_event_tx.send(AppEvent::Error {
                message: format!("Failed to submit: {}", err),
            }).ok();
        }
    }
}
```

### Poll Task

```rust
fn spawn_poll_task(
    client: Arc<Client>,
    job_id: String,
    app_event_tx: UnboundedSender<AppEvent>,
) {
    tokio::spawn(async move {
        let mut delay = Duration::from_millis(400);  // Start at 400ms
        let max_delay = Duration::from_secs(2);      // Cap at 2s

        loop {
            tokio::time::sleep(delay).await;

            // Exponential backoff
            delay = (delay * 2).min(max_delay);

            // Poll status
            match client.get_status(&job_id).await {
                Ok(status) => {
                    let state = status.state.clone();

                    // Send update event
                    app_event_tx.send(AppEvent::JobUpdated {
                        job_id: job_id.clone(),
                        status: status.clone(),
                    }).ok();

                    // Check if complete
                    match state {
                        JobState::Succeeded => {
                            // Fetch artifact
                            if let Ok(artifact) = client.get_artifact(&job_id).await {
                                // Persist to local disk
                                let local_path = persist_artifact(&artifact).await;

                                app_event_tx.send(AppEvent::JobCompleted {
                                    job_id: job_id.clone(),
                                    artifact,
                                }).ok();

                                // Auto-play
                                app_event_tx.send(AppEvent::PlaybackStarted {
                                    job_id: job_id.clone(),
                                }).ok();
                            }
                            break;
                        }

                        JobState::Failed => {
                            break;
                        }

                        _ => {
                            // Keep polling
                        }
                    }
                }

                Err(err) => {
                    app_event_tx.send(AppEvent::Error {
                        message: format!("Poll error: {}", err),
                    }).ok();
                    break;
                }
            }
        }
    });
}
```

---

## 5. UI Rendering (`ui/mod.rs`)

### Main Render Function

```rust
pub fn render(f: &mut Frame, app: &mut AppState) {
    // Split screen: 68% left, 32% right
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(68),
            Constraint::Percentage(32),
        ])
        .split(f.size());

    // Left side: conversation + prompt
    render_left_panel(f, app, chunks[0]);

    // Right side: jobs + backends + status
    render_right_panel(f, app, chunks[1]);
}
```

### Left Panel (Conversation + Prompt)

```rust
fn render_left_panel(f: &mut Frame, app: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),      // Status bar
            Constraint::Min(0),         // Conversation
            Constraint::Length(3),      // Prompt input
        ])
        .split(area);

    // Status bar
    render_status_bar(f, app, chunks[0]);

    // Conversation history
    render_conversation(f, app, chunks[1]);

    // Prompt input
    render_prompt_input(f, app, chunks[2]);
}
```

### Status Bar

```rust
fn render_status_bar(f: &mut Frame, app: &AppState, area: Rect) {
    // Find active job
    let active_job = app.jobs.values()
        .find(|j| matches!(j.status.state, JobState::Running));

    let text = if let Some(job) = active_job {
        // Show progress
        let progress = (job.status.progress * 100.0) as u8;
        format!(
            "Progress: [{}{}] {}% • {}",
            "█".repeat(progress as usize / 5),
            "░".repeat(20 - (progress as usize / 5)),
            progress,
            job.status.message
        )
    } else {
        "Ready".to_string()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let paragraph = Paragraph::new(text)
        .block(block)
        .style(Style::default().fg(Color::White));

    f.render_widget(paragraph, area);
}
```

### Conversation Pane

```rust
fn render_conversation(f: &mut Frame, app: &mut AppState, area: Rect) {
    let messages: Vec<ListItem> = app.chat.iter()
        .map(|entry| {
            let timestamp = entry.timestamp.format("%H:%M:%S");
            let role_style = match entry.role {
                ChatRole::User => Style::default().fg(Color::Green),
                ChatRole::Worker => Style::default().fg(Color::Cyan),
                ChatRole::System => Style::default().fg(Color::Yellow),
            };

            let content = format!(
                "[{}] {}: {}",
                timestamp,
                entry.role,
                entry.content
            );

            ListItem::new(content).style(role_style)
        })
        .collect();

    let list = List::new(messages)
        .block(Block::default()
            .title("Conversation")
            .borders(Borders::ALL))
        .highlight_style(Style::default().bg(Color::DarkGray));

    // Auto-scroll if following
    let mut state = ListState::default();
    if app.chat_following && !app.chat.is_empty() {
        state.select(Some(app.chat.len() - 1));
    }

    f.render_stateful_widget(list, area, &mut state);
}
```

### Right Panel (Jobs + Backends + Status)

```rust
fn render_right_panel(f: &mut Frame, app: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),  // Jobs
            Constraint::Percentage(20),  // Backends
            Constraint::Percentage(30),  // Status
        ])
        .split(area);

    render_jobs_pane(f, app, chunks[0]);
    render_backends_pane(f, app, chunks[1]);
    render_status_pane(f, app, chunks[2]);
}
```

### Jobs Pane

```rust
fn render_jobs_pane(f: &mut Frame, app: &AppState, area: Rect) {
    let items: Vec<ListItem> = app.jobs.iter()
        .map(|(job_id, entry)| {
            let state_symbol = match entry.status.state {
                JobState::Queued => "⏳",
                JobState::Running => "▶",
                JobState::Succeeded => "✓",
                JobState::Failed => "✗",
            };

            let state_color = match entry.status.state {
                JobState::Queued => Color::Yellow,
                JobState::Running => Color::Cyan,
                JobState::Succeeded => Color::Green,
                JobState::Failed => Color::Red,
            };

            let progress = if entry.status.state == JobState::Running {
                format!(" {}%", (entry.status.progress * 100.0) as u8)
            } else {
                String::new()
            };

            let content = format!(
                "{} {}{}",
                state_symbol,
                &job_id[..8],
                progress
            );

            ListItem::new(content).style(Style::default().fg(state_color))
        })
        .collect();

    let list = List::new(items)
        .block(Block::default()
            .title("Jobs")
            .borders(Borders::ALL))
        .highlight_style(Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD));

    let mut state = ListState::default();
    state.select(app.selected_job_index);

    f.render_stateful_widget(list, area, &mut state);
}
```

---

## 6. HTTP Client (`api.rs`)

### Client Structure

```rust
pub struct Client {
    client: reqwest::Client,
    base_url: String,
}

impl Client {
    pub fn new(base_url: &str) -> Arc<Self> {
        Arc::new(Self {
            client: reqwest::Client::builder()
                .use_rustls_tls()
                .build()
                .unwrap(),
            base_url: base_url.to_string(),
        })
    }

    pub async fn health(&self) -> Result<HealthResponse> {
        let url = format!("{}/health", self.base_url);
        let response = self.client.get(&url).send().await?;
        Ok(response.json().await?)
    }

    pub async fn submit_generation(&self, request: GenerationRequest) -> Result<GenerationStatus> {
        let url = format!("{}/generate", self.base_url);
        let response = self.client.post(&url)
            .json(&request)
            .send()
            .await?;
        Ok(response.json().await?)
    }

    pub async fn get_status(&self, job_id: &str) -> Result<GenerationStatus> {
        let url = format!("{}/status/{}", self.base_url, job_id);
        let response = self.client.get(&url).send().await?;
        Ok(response.json().await?)
    }

    pub async fn get_artifact(&self, job_id: &str) -> Result<GenerationArtifact> {
        let url = format!("{}/artifact/{}", self.base_url, job_id);
        let response = self.client.get(&url).send().await?;
        Ok(response.json().await?)
    }
}
```

---

## 7. Audio Playback

### AudioPlayer (rodio wrapper)

```rust
pub struct AudioPlayer {
    _stream: OutputStream,
    sink: Arc<Mutex<Sink>>,
}

impl AudioPlayer {
    pub fn new() -> Result<Self> {
        let (stream, stream_handle) = OutputStream::try_default()?;
        let sink = Sink::try_new(&stream_handle)?;

        Ok(Self {
            _stream: stream,
            sink: Arc::new(Mutex::new(sink)),
        })
    }

    pub fn play(&self, path: &Path) -> Result<()> {
        let file = File::open(path)?;
        let source = Decoder::new(BufReader::new(file))?;

        let sink = self.sink.lock().unwrap();
        sink.append(source);
        sink.play();

        Ok(())
    }

    pub fn stop(&self) {
        let sink = self.sink.lock().unwrap();
        sink.stop();
    }

    pub fn is_playing(&self) -> bool {
        let sink = self.sink.lock().unwrap();
        !sink.empty()
    }
}

// Thread safety (rodio Sink is NOT Send, but we wrap it)
unsafe impl Send for AudioPlayer {}
unsafe impl Sync for AudioPlayer {}
```

---

## 8. Configuration (`config.rs`)

### Config Loading

```rust
pub struct Config {
    pub worker_url: String,
    pub default_model: String,
    pub default_duration: u8,
    pub artifact_dir: PathBuf,
}

impl Config {
    pub fn load() -> Result<Self> {
        // 1. Start with defaults
        let mut config = Self::default();

        // 2. Load from TOML file if exists
        let config_path = Self::config_file_path();
        if config_path.exists() {
            let toml_str = fs::read_to_string(&config_path)?;
            let toml_config: TomlConfig = toml::from_str(&toml_str)?;
            config.merge_toml(toml_config);
        }

        // 3. Override with environment variables
        if let Ok(url) = env::var("TIMBRE_WORKER_URL") {
            config.worker_url = url;
        }
        if let Ok(model) = env::var("TIMBRE_DEFAULT_MODEL") {
            config.default_model = model;
        }
        if let Ok(duration) = env::var("TIMBRE_DEFAULT_DURATION") {
            if let Ok(d) = duration.parse() {
                config.default_duration = d;
            }
        }

        Ok(config)
    }

    fn config_file_path() -> PathBuf {
        // XDG standard: ~/.config/timbre/config.toml
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("timbre")
            .join("config.toml")
    }
}
```

---

## 9. Key Bindings

### Normal Mode
- `q` / `Ctrl+C`: Quit
- `i`: Enter Insert mode
- `Esc`: Clear input, ensure Normal mode
- `Tab`: Cycle panes forward
- `Shift+Tab`: Cycle panes backward
- `↑` / `↓`: Navigate (scroll or select job)
- `←` / `→`: Switch panes
- `Ctrl+P`: Play selected job
- `s` (in Status pane): Stop playback

### Insert Mode
- `Type`: Add characters to input
- `Backspace`: Delete character
- `Enter`: Submit prompt/command
- `Esc`: Exit Insert mode

---

## 10. Best Practices

### For Performance
- Use `select!` macro for async event handling
- Batch UI updates (don't render on every event)
- Use IndexMap for insertion-order jobs
- Cache computed UI elements

### For Responsiveness
- Never block in event loop
- Spawn background tasks for I/O
- Use channels for async communication
- Keep render function fast

### For Debugging
- Use `tracing` crate for structured logging
- Log all HTTP requests/responses
- Track event flow with debug logs
- Capture panics gracefully

---

## References

- **Ratatui:** https://github.com/ratatui-org/ratatui
- **Crossterm:** https://github.com/crossterm-rs/crossterm
- **Tokio:** https://tokio.rs
- **Rodio:** https://github.com/RustAudio/rodio

---

**Document Version:** 1.0
**Last Updated:** 2025
**Ratatui Version:** 0.26+
**Architecture:** Event-driven async TUI
