use crate::{
    app::{
        format_request_summary, AppCommand, AppEvent, AppState, FocusedPane, GenerationOverrides,
        InputMode, JobEntry, SessionCliCommand,
    },
    session::{clip_layer_label, ClipSlotStatus, SessionScene, LAYER_ORDER},
    types::{
        ClipLayer, CompositionSection, GenerationMode, JobState, SectionEnergy,
        SectionOrchestration, SessionCreateRequest,
    },
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table, Wrap},
    Terminal,
};
use serde::Deserialize;
use std::time::Duration;
use std::{collections::HashMap, time::Duration as StdDuration};
use tokio::sync::mpsc::{error::TryRecvError, UnboundedReceiver, UnboundedSender};

pub fn run<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut AppState,
    event_rx: &mut UnboundedReceiver<AppEvent>,
    command_tx: UnboundedSender<AppCommand>,
) -> Result<()> {
    loop {
        drain_events(app, event_rx);
        terminal.draw(|frame| draw_ui(frame, app))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if handle_key(app, &command_tx, key)? {
                    break;
                }
            }
        }
    }

    Ok(())
}

fn drain_events(app: &mut AppState, event_rx: &mut UnboundedReceiver<AppEvent>) {
    loop {
        match event_rx.try_recv() {
            Ok(event) => app.handle_event(event),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => break,
        }
    }
}

fn draw_ui(frame: &mut ratatui::Frame, app: &AppState) {
    let layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
        .split(frame.size());

    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(1)])
        .split(layout[0]);

    render_status_bar(frame, body[0], app);

    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(3)])
        .split(body[1]);

    render_session_view(frame, left[0], app);
    render_prompt(frame, left[1], app);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(20),
            Constraint::Percentage(30),
        ])
        .split(layout[1]);

    render_jobs(frame, right[0], app);
    render_backend_diagnostics(frame, right[1], app);
    render_status(frame, right[2], app);
}

fn render_session_view(frame: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let scenes = app.session_view.scenes();
    let mut header_cells: Vec<Cell> = Vec::new();
    header_cells.push(Cell::from(Span::styled(
        "Layer",
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )));
    if scenes.is_empty() {
        header_cells.push(Cell::from("Scene 1"));
    } else {
        for scene in scenes {
            header_cells.push(Cell::from(scene.name.clone()));
        }
    }

    let header = Row::new(header_cells).height(1).bottom_margin(0);

    let (focused_layer, focused_scene) = app.session_view.focused();

    let mut rows: Vec<Row> = Vec::new();
    let display_scenes = if scenes.is_empty() {
        vec![SessionScene { index: 0, name: "Scene 1".to_string() }]
    } else {
        scenes.to_vec()
    };

    for layer in LAYER_ORDER {
        let mut cells: Vec<Cell> = Vec::new();
        let layer_label = clip_layer_label(layer);
        let mut label_style = Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD);
        if focused_layer == layer {
            label_style = label_style.add_modifier(Modifier::REVERSED);
        }
        cells.push(Cell::from(Span::styled(layer_label, label_style)));

        for scene in &display_scenes {
            let slot = app.session_view.slot(layer, scene.index);
            let mut spans: Vec<Span> = Vec::new();
            let playing = app.session_view.layer_playing(layer) == Some(scene.index);
            if playing {
                spans.push(Span::styled(
                    "▶ ",
                    Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD),
                ));
            }
            let primary_text = slot
                .prompt
                .as_deref()
                .filter(|text| !text.is_empty())
                .map(|text| truncate_text(text, 18))
                .unwrap_or_else(|| "—".to_string());
            let mut cell_style = match slot.status {
                ClipSlotStatus::Empty => Style::default().fg(Color::DarkGray),
                ClipSlotStatus::Pending => Style::default().fg(Color::Blue),
                ClipSlotStatus::Rendering => Style::default().fg(Color::LightBlue),
                ClipSlotStatus::Ready => Style::default().fg(Color::Green),
                ClipSlotStatus::Failed => Style::default().fg(Color::Red),
            };
            if focused_layer == layer && focused_scene == scene.index {
                cell_style = cell_style.add_modifier(Modifier::BOLD);
            }
            spans.push(Span::styled(primary_text, cell_style));

            let status_text = match slot.status {
                ClipSlotStatus::Empty => "".to_string(),
                ClipSlotStatus::Pending => "(queued)".to_string(),
                ClipSlotStatus::Rendering => "(rendering)".to_string(),
                ClipSlotStatus::Ready => {
                    let bars = slot.bars.map(|b| format!("{b} bars")).unwrap_or_default();
                    bars
                }
                ClipSlotStatus::Failed => "(failed)".to_string(),
            };
            let mut lines = vec![Line::from(spans)];
            if !status_text.is_empty() {
                lines.push(Line::from(Span::styled(
                    status_text,
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                )));
            }
            cells.push(Cell::from(lines));
        }

        rows.push(Row::new(cells).height(2));
    }

    let mut block = Block::default().title("Session").borders(Borders::ALL);
    if app.focused_pane == FocusedPane::SessionView {
        block = block.border_style(pane_border(app, FocusedPane::SessionView));
    }

    let widths: Vec<Constraint> = std::iter::once(Constraint::Length(12))
        .chain(
            std::iter::repeat(Constraint::Ratio(1, (display_scenes.len().max(1)) as u32))
                .take(display_scenes.len()),
        )
        .collect();

    let table = Table::new(rows, widths).header(header).block(block).column_spacing(1);

    frame.render_widget(table, area);
}

fn render_prompt(frame: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let prompt_title = app.prompt_panel_title();
    let mut block = Block::default().title(prompt_title).borders(Borders::ALL);
    if app.focused_pane == FocusedPane::Prompt {
        block = block.border_style(pane_border(app, FocusedPane::Prompt));
    } else {
        block = block.border_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    }
    let paragraph = Paragraph::new(app.input.clone()).block(block);
    frame.render_widget(paragraph, area);
}

fn render_jobs(frame: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let items: Vec<ListItem> =
        app.jobs_iter().rev().map(|(job_id, job)| job_list_item(app, job_id, job)).collect();

    let mut block = Block::default().title("Jobs").borders(Borders::ALL);
    if app.focused_pane == FocusedPane::Jobs {
        block = block.border_style(pane_border(app, FocusedPane::Jobs));
    }

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));
    frame.render_widget(list, area);
}

fn render_backend_diagnostics(frame: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let mut lines: Vec<Line> = Vec::new();

    if app.backend_status.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "Awaiting backend telemetry…",
            Style::default().fg(Color::DarkGray),
        )]));
    } else {
        for status in &app.backend_status {
            let style = if status.ready {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            };
            let prefix = if status.ready { "✔" } else { "!" };
            lines.push(Line::from(vec![
                Span::styled(prefix, style),
                Span::raw(" "),
                Span::styled(status.name.clone(), style),
                Span::raw(" "),
                Span::styled(status.summary(), Style::default().fg(Color::Gray)),
            ]));
            for (key, value) in &status.details {
                lines.push(Line::from(vec![Span::styled(
                    format!("  {key}: {value}"),
                    Style::default().fg(Color::DarkGray),
                )]));
            }
            if let Some(error) = &status.error {
                lines.push(Line::from(vec![Span::styled(
                    format!("  error: {error}"),
                    Style::default().fg(Color::Red),
                )]));
            }
            if let Some(updated_at) = &status.updated_at {
                lines.push(Line::from(vec![Span::styled(
                    format!("  updated {updated_at}"),
                    Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
                )]));
            }
            lines.push(Line::default());
        }
        if lines.last().map(|line| line.spans.is_empty()).unwrap_or(false) {
            lines.pop();
        }
    }

    if let Some((_id, job)) = app.selected_job() {
        if let Some(summary) = conditioning_summary(job) {
            lines.push(Line::default());
            lines.push(Line::from(vec![Span::styled(
                format!(
                    "Conditioning: {} sections (avg tail {:.2}s · max tail {:.2}s · avg pad {:.2}s)",
                    summary.phrase_sections,
                    summary.avg_tail,
                    summary.max_tail,
                    summary.avg_padding,
                ),
                Style::default().fg(Color::LightYellow),
            )]));
            if summary.missing_tails > 0 {
                lines.push(Line::from(vec![Span::styled(
                    format!("  missing tails: {}", summary.missing_tails),
                    Style::default().fg(Color::Magenta),
                )]));
            }
        }
    }

    if lines.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "No backend data yet.",
            Style::default().fg(Color::Gray),
        )]));
    }

    let mut block = Block::default().title("Backends").borders(Borders::ALL);
    if matches!(app.focused_pane, FocusedPane::Status) {
        block = block.border_style(pane_border(app, FocusedPane::Status));
    }
    let paragraph = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn job_list_item(app: &AppState, job_id: &str, job: &JobEntry) -> ListItem<'static> {
    let marker = if app.focused_job.as_deref() == Some(job_id) {
        Span::styled("▶ ", Style::default().fg(Color::Yellow))
    } else {
        Span::raw("  ")
    };

    let progress = (job.status.progress * 100.0).round().clamp(0.0, 100.0) as i32;
    let job_label = job_id.split('-').next().unwrap_or(job_id);
    let mut spans = vec![
        marker,
        Span::styled(
            format!("{job_label:<8}"),
            Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
        ),
        Span::styled(
            job.state_label(),
            match job.status.state {
                JobState::Queued => Style::default().fg(Color::Blue),
                JobState::Running => Style::default().fg(Color::Cyan),
                JobState::Succeeded => Style::default().fg(Color::Green),
                JobState::Failed => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            },
        ),
        Span::raw(format!(" {:>3}% ", progress)),
        Span::raw(truncate_text(&job.prompt, 34)),
    ];

    let summary = truncate_text(&format_request_summary(&job.request), 28);
    spans.push(Span::raw(" · "));
    spans.push(Span::styled(summary, Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM)));

    if let Some(message) = job.status.message.as_deref() {
        spans.push(Span::raw(" · "));
        spans.push(Span::styled(truncate_text(message, 30), Style::default().fg(Color::Gray)));
    } else if let Some(artifact) = &job.artifact {
        spans.push(Span::raw(" · "));
        spans.push(Span::styled(
            artifact.local_path.display().to_string(),
            Style::default().fg(Color::Green),
        ));
        if let Some(extras) = artifact.descriptor.metadata.extras.as_object() {
            if extras.get("placeholder").and_then(|value| value.as_bool()).unwrap_or(false) {
                spans.push(Span::raw(" · "));
                spans.push(Span::styled("placeholder", Style::default().fg(Color::Magenta)));
                if let Some(reason) =
                    extras.get("placeholder_reason").and_then(|value| value.as_str())
                {
                    spans.push(Span::raw(" · "));
                    spans.push(Span::styled(
                        truncate_text(reason, 18),
                        Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
                    ));
                }
            } else if let Some(backend) = extras.get("backend").and_then(|value| value.as_str()) {
                spans.push(Span::raw(" · "));
                spans.push(Span::styled(
                    truncate_text(backend, 16),
                    Style::default().fg(Color::Green).add_modifier(Modifier::DIM),
                ));
            }
            if let Some(hash) = extras.get("prompt_hash").and_then(|value| value.as_str()) {
                spans.push(Span::raw(" · "));
                spans.push(Span::styled(
                    format!("hash {}", truncate_text(hash, 12)),
                    Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
                ));
            }
        }
    }

    ListItem::new(Line::from(spans))
}

fn render_status(frame: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let mut lines: Vec<Line> = Vec::new();

    if let Some((ratio, elapsed, duration, playing, job_id, path)) = app.playback_progress() {
        let status = if playing { "Playing" } else { "Playback paused" };
        lines.push(Line::from(vec![
            Span::styled(
                format!("{}: {}", status, path.display()),
                Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(" (job {})", job_id)),
        ]));
        lines.push(Line::from(vec![Span::styled(
            format!("Progress: {}/{}", format_duration(elapsed), format_duration(duration)),
            Style::default().fg(Color::LightBlue),
        )]));

        let bar_width = area.width.saturating_sub(4);
        let filled = (bar_width as f32 * ratio).round() as u16;
        let mut spans = Vec::new();
        spans.push(Span::raw("  "));
        for idx in 0..bar_width {
            if idx < filled {
                spans.push(Span::styled("█", glow_style()));
            } else {
                spans.push(Span::styled("░", Style::default().fg(Color::DarkGray)));
            }
        }
        lines.push(Line::from(spans));
        lines.push(Line::default());
    }

    lines.push(Line::from(vec![Span::raw(format!("Config: {}", app.config_summary()))]));
    if let Some(session) = &app.session {
        let label = AppState::format_session_label(session);
        let tempo = session
            .tempo_bpm
            .map(|bpm| format!("{bpm} BPM"))
            .unwrap_or_else(|| "tempo pending".to_string());
        let key = session.key.as_deref().unwrap_or("key pending");
        lines.push(Line::from(vec![Span::styled(
            format!("Session {label} · {tempo} · {key} · {} clips", session.clip_count),
            Style::default().fg(Color::LightCyan),
        )]));
        if let Some(seed_prompt) = &session.seed_prompt {
            lines.push(Line::from(vec![Span::styled(
                format!("  Seed prompt: {}", truncate_text(seed_prompt, 60)),
                Style::default().fg(Color::Gray),
            )]));
        }
    }
    if let Some(model) = app.last_submission_model() {
        lines.push(Line::from(vec![Span::styled(
            format!("Last render: {model}"),
            Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD),
        )]));
    }
    if app.status_lines.is_empty() {
        lines.push(Line::from(vec![Span::raw("No activity yet.")]))
    } else {
        let start = app.status_scroll.min(app.status_lines.len());
        for entry in app.status_lines.iter().skip(start) {
            lines.push(Line::from(vec![Span::raw(entry.clone())]));
        }
    }

    if let Some((_id, job)) = app.selected_job() {
        lines.push(Line::default());
        lines.extend(plan_status_lines(job));
    }

    let mut block = Block::default().title("Status").borders(Borders::ALL);
    if app.focused_pane == FocusedPane::Status {
        block = block.border_style(pane_border(app, FocusedPane::Status));
    }
    let total_lines = lines.len();
    let height = area.height as usize;
    let max_offset = total_lines.saturating_sub(height);
    let desired = if app.status_following {
        max_offset
    } else {
        max_offset.saturating_sub(app.status_scroll.min(max_offset))
    };
    let paragraph = Paragraph::new(lines).block(block).scroll((desired as u16, 0));
    frame.render_widget(paragraph, area);
}

fn render_status_bar(frame: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let now = Utc::now();
    let mut base_progress = app.status_bar.progress.clamp(0.0, 1.0);
    let mut message = app.status_bar.message.clone();
    let mut glow_source = app.status_bar.last_update;

    if let Some((ratio, elapsed, duration, playing, job_id, path)) = app.playback_progress() {
        base_progress = ratio;
        glow_source = app.status_bar.last_update;
        message = if playing {
            format!(
                " Playing {} (job {}) — {}/{} ",
                path.display(),
                job_id,
                format_duration(elapsed),
                format_duration(duration)
            )
        } else {
            format!(
                " Playback paused (job {}) — {}/{} ",
                job_id,
                format_duration(elapsed),
                format_duration(duration)
            )
        };
    }

    let filled_width = (area.width as f32 * base_progress).round() as u16;
    let glow = glow_intensity(glow_source, now);

    let mut lines = Vec::new();
    let info_line = Line::from(vec![Span::styled(
        message,
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD)
            .add_modifier(if glow > 0.5 { Modifier::ITALIC } else { Modifier::empty() }),
    )]);
    lines.push(info_line);

    let bar_line = (0..area.width)
        .map(|x| {
            if x < filled_width {
                Span::styled("█", Style::default().fg(glow_color(glow)))
            } else {
                Span::styled("░", Style::default().fg(Color::DarkGray))
            }
        })
        .collect::<Vec<_>>();
    lines.push(Line::from(bar_line));

    let mut block = Block::default().borders(Borders::ALL).title("Status");
    if app.focused_pane == FocusedPane::StatusBar {
        block = block.border_style(pane_border(app, FocusedPane::StatusBar));
    }
    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}

fn handle_key(
    app: &mut AppState,
    command_tx: &UnboundedSender<AppCommand>,
    key: crossterm::event::KeyEvent,
) -> Result<bool> {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(true),
        KeyCode::Char('q') if matches!(app.input_mode, InputMode::Normal) => return Ok(true),
        KeyCode::Char('s') if matches!(app.focused_pane, FocusedPane::Status) => {
            let _ = command_tx.send(AppCommand::StopPlayback);
        }
        KeyCode::Char('i') if matches!(app.input_mode, InputMode::Normal) => {
            app.input_mode = InputMode::Insert;
            if matches!(app.focused_pane, FocusedPane::Status) {
                app.status_following = false;
            }
        }
        KeyCode::Esc => {
            if app.input_mode == InputMode::Insert {
                app.input_mode = InputMode::Normal;
            }
            if matches!(app.focused_pane, FocusedPane::Status) {
                app.status_scroll = 0;
                app.status_following = true;
            }
            if matches!(app.focused_pane, FocusedPane::Prompt) {
                app.input.clear();
            }
        }
        KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if let Some((job_id, job)) = app.selected_job() {
                if job.artifact.is_some() {
                    let _ = command_tx.send(AppCommand::PlayJob { job_id: job_id.clone() });
                } else {
                    app.push_status_line(format!("Job {} has no artifact yet", job_id));
                }
            }
        }
        KeyCode::Up => {
            if matches!(app.input_mode, InputMode::Insert) {
                match app.focused_pane {
                    FocusedPane::Status => app.increment_status_scroll(1),
                    FocusedPane::Jobs => app.select_previous_job(),
                    _ => {}
                }
            } else {
                match app.focused_pane {
                    FocusedPane::Prompt => app.focused_pane = FocusedPane::SessionView,
                    FocusedPane::SessionView => app.session_view.focus_prev_layer(),
                    FocusedPane::Jobs => app.focused_pane = FocusedPane::SessionView,
                    FocusedPane::Status => app.focused_pane = FocusedPane::Jobs,
                    FocusedPane::StatusBar => app.focused_pane = FocusedPane::SessionView,
                }
            }
        }
        KeyCode::Down => {
            if matches!(app.input_mode, InputMode::Insert) {
                match app.focused_pane {
                    FocusedPane::Status => app.increment_status_scroll(-1),
                    FocusedPane::Jobs => app.select_next_job(),
                    _ => {}
                }
            } else {
                match app.focused_pane {
                    FocusedPane::StatusBar => app.focused_pane = FocusedPane::SessionView,
                    FocusedPane::SessionView => app.session_view.focus_next_layer(),
                    FocusedPane::Prompt => app.focused_pane = FocusedPane::Status,
                    FocusedPane::Jobs => app.focused_pane = FocusedPane::Status,
                    _ => {}
                }
            }
        }
        KeyCode::Left if matches!(app.input_mode, InputMode::Normal) => match app.focused_pane {
            FocusedPane::SessionView => app.session_view.focus_prev_scene(),
            FocusedPane::Prompt => app.focused_pane = FocusedPane::Jobs,
            FocusedPane::Status => app.focused_pane = FocusedPane::SessionView,
            FocusedPane::Jobs => app.focused_pane = FocusedPane::Status,
            FocusedPane::StatusBar => app.focused_pane = FocusedPane::Status,
        },
        KeyCode::Right if matches!(app.input_mode, InputMode::Normal) => match app.focused_pane {
            FocusedPane::SessionView => app.session_view.focus_next_scene(),
            FocusedPane::Prompt => app.focused_pane = FocusedPane::Jobs,
            FocusedPane::Status => app.focused_pane = FocusedPane::SessionView,
            FocusedPane::Jobs => app.focused_pane = FocusedPane::Prompt,
            FocusedPane::StatusBar => app.focused_pane = FocusedPane::Status,
        },
        KeyCode::Tab if matches!(app.input_mode, InputMode::Normal) => app.focus_next(),
        KeyCode::BackTab if matches!(app.input_mode, InputMode::Normal) => app.focus_previous(),
        KeyCode::Backspace
            if matches!(app.focused_pane, FocusedPane::Prompt)
                && matches!(app.input_mode, InputMode::Insert) =>
        {
            app.input.pop();
        }
        KeyCode::Enter
            if matches!(app.input_mode, InputMode::Normal)
                && matches!(app.focused_pane, FocusedPane::SessionView) =>
        {
            if let Some(context) = app.clip_launch_context() {
                let _ = command_tx.send(AppCommand::LaunchClip {
                    layer: context.layer,
                    scene_index: context.scene_index,
                    path: context.path,
                    tempo_bpm: context.tempo_bpm,
                    beats_per_bar: context.beats_per_bar,
                });
            } else {
                app.push_status_line("Clip not ready yet".to_string());
            }
        }
        KeyCode::Enter
            if matches!(app.focused_pane, FocusedPane::Prompt)
                && matches!(app.input_mode, InputMode::Insert) =>
        {
            let line = app.input.trim().to_string();
            if line.is_empty() {
                app.input.clear();
                return Ok(false);
            }

            let mut prompt_text = line.clone();
            let mut inline_overrides: Option<GenerationOverrides> = None;

            if let Some(stripped) = line.strip_prefix('/') {
                let trimmed = stripped.trim_start();
                if trimmed.is_empty() {
                    dispatch_app_command(app, &command_tx, &line);
                    app.input.clear();
                    return Ok(false);
                }

                if let Some(command_token) = trimmed.split_whitespace().next() {
                    let command_lower = command_token.to_ascii_lowercase();
                    let rest = trimmed[command_token.len()..].trim_start();

                    if let Some(inline) = app.inline_command_for(&command_lower) {
                        if rest.is_empty() {
                            let message = inline.usage_hint.to_string();
                            app.push_status_line(message);
                            app.input.clear();
                            return Ok(false);
                        }
                        inline_overrides = Some(inline.overrides);
                        prompt_text = rest.to_string();
                    } else {
                        dispatch_app_command(app, &command_tx, &line);
                        app.input.clear();
                        return Ok(false);
                    }
                } else {
                    app.input.clear();
                    return Ok(false);
                }
            }

            if prompt_text.is_empty() {
                app.input.clear();
                return Ok(false);
            }

            let prompt = prompt_text;
            let overrides_ref = inline_overrides.as_ref();
            let (request, plan) =
                app.build_generation_payload_with_overrides(&prompt, overrides_ref);
            if let Some(overrides) = overrides_ref {
                if matches!(overrides.mode, Some(GenerationMode::Motif)) {
                    app.push_status_line("Queued motif preview (~16s)".to_string());
                }
            }
            if command_tx.send(AppCommand::SubmitPrompt { prompt, request, plan }).is_err() {
                app.push_status_line("Failed to submit prompt; worker channel closed".into());
            }
            app.input.clear();
            app.focused_pane = FocusedPane::StatusBar;
            app.status_scroll = 0;
            app.status_following = true;
            app.input_mode = InputMode::Normal;
        }
        KeyCode::Char(' ')
            if matches!(app.input_mode, InputMode::Normal)
                && matches!(app.focused_pane, FocusedPane::SessionView) =>
        {
            let layer = app.focused_clip_layer();
            let _ = command_tx.send(AppCommand::StopClip { layer });
        }
        KeyCode::Char('x') if matches!(app.input_mode, InputMode::Normal) => {
            let _ = command_tx.send(AppCommand::StopAllClips);
        }
        KeyCode::Char(c)
            if matches!(app.focused_pane, FocusedPane::Prompt)
                && matches!(app.input_mode, InputMode::Insert) =>
        {
            if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                app.input.push(c);
            }
        }
        _ => {}
    }

    Ok(false)
}

fn dispatch_app_command(
    app: &mut AppState,
    command_tx: &UnboundedSender<AppCommand>,
    command_line: &str,
) {
    let trimmed = command_line.trim();
    if trimmed.starts_with("/clip") {
        let mut parts = trimmed.split_whitespace();
        let _ = parts.next();
        let Some(layer_token) = parts.next() else {
            let message = "Usage: /clip <layer> [prompt]".to_string();
            app.push_status_line(message);
            return;
        };
        let Some(layer) = parse_clip_layer(layer_token) else {
            let message = format!(
                "Unknown clip layer `{}`. Use rhythm, bass, harmony, lead, textures, or vocals.",
                layer_token
            );
            app.push_status_line(message);
            return;
        };
        app.session_view.focus_layer(layer);
        app.session_view.focus_layer(layer);
        let prompt_tail = parts.collect::<Vec<_>>().join(" ");
        let prompt_override = if prompt_tail.is_empty() { None } else { Some(prompt_tail) };
        match app.build_clip_payload(layer, prompt_override) {
            Ok((request, plan, prompt_text)) => {
                let label = clip_layer_label(layer);
                if command_tx
                    .send(AppCommand::SubmitClip { prompt: prompt_text, layer, request, plan })
                    .is_err()
                {
                    let message =
                        "Failed to submit clip request; worker channel closed".to_string();
                    app.push_status_line(message);
                } else {
                    let (_, scene_index) = app.session_view.focused();
                    let scene_name = app.session_view.scene_name(scene_index);
                    app.push_status_line(format!("Queued clip for {label} · {scene_name}"));
                }
            }
            Err(err) => {
                app.push_status_line(err);
            }
        }
        return;
    }
    if trimmed.starts_with("/scene") {
        let mut parts = trimmed.split_whitespace();
        let _ = parts.next();
        match parts.next() {
            Some("add") => {
                let name = parts.collect::<Vec<_>>().join(" ");
                let new_index = app.session_view.add_scene_after_focus();
                if !name.trim().is_empty() {
                    app.session_view.rename_scene(new_index, name.clone());
                }
                app.session_view.focus_scene(new_index);
                let scene_name = app.session_view.scene_name(new_index);
                app.push_status_line(format!("Added {scene_name}"));
            }
            Some("rename") => {
                let scene_index = app.session_view.focused().1;
                let new_name = parts.collect::<Vec<_>>().join(" ");
                if new_name.trim().is_empty() {
                    app.push_status_line("Usage: /scene rename <name>".to_string());
                } else {
                    app.session_view.rename_scene(scene_index, new_name.clone());
                    app.push_status_line(format!(
                        "Scene {} renamed",
                        app.session_view.scene_name(scene_index)
                    ));
                }
            }
            Some(other) => {
                let message = format!("Unknown scene command `{other}`");
                app.push_status_line(message);
            }
            None => {
                let message = "Usage: /scene add [name] | /scene rename <name>".to_string();
                app.push_status_line(message);
            }
        }
        return;
    }
    if trimmed.starts_with("/session") {
        match app.parse_session_command(command_line) {
            Ok(SessionCliCommand::Start) => {
                let payload = SessionCreateRequest::default();
                if command_tx.send(AppCommand::CreateSession { payload }).is_err() {
                    let message =
                        "Failed to request session creation; worker channel closed".to_string();
                    app.push_status_line(message);
                } else {
                    let message = "Requesting new session from worker…".to_string();
                    app.push_status_line(message);
                }
            }
            Ok(SessionCliCommand::Status) => {
                if let Some(session_id) = app.current_session_id() {
                    if command_tx
                        .send(AppCommand::FetchSession { session_id: session_id.clone() })
                        .is_err()
                    {
                        let message =
                            "Failed to refresh session status; worker channel closed".to_string();
                        app.push_status_line(message);
                    } else {
                        app.push_status_line("Refreshing session status…".to_string());
                    }
                } else {
                    let message = "No active session. Use /session start before requesting status."
                        .to_string();
                    app.push_status_line(message);
                }
            }
            Err(error) => {
                if error != "not a session command" {
                    let message = format!("Session command error: {error}");
                    app.push_status_line(message);
                }
            }
        }
        return;
    }

    match app.handle_command(command_line) {
        Ok(message) => {
            app.push_status_line(message);
        }
        Err(err) => {
            let message = format!("Command error: {err}");
            app.push_status_line(message);
        }
    }
}

fn parse_clip_layer(token: &str) -> Option<ClipLayer> {
    match token.to_ascii_lowercase().as_str() {
        "rhythm" => Some(ClipLayer::Rhythm),
        "bass" => Some(ClipLayer::Bass),
        "harmony" => Some(ClipLayer::Harmony),
        "lead" => Some(ClipLayer::Lead),
        "textures" => Some(ClipLayer::Textures),
        "vocals" => Some(ClipLayer::Vocals),
        _ => None,
    }
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let truncated: String = text.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{}…", truncated)
    }
}

fn plan_status_lines(job: &JobEntry) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    lines.push(Line::from(vec![Span::styled(
        format!(
            "Plan: {} sections · {} BPM · {}",
            job.plan.sections.len(),
            job.plan.tempo_bpm,
            job.plan.key
        ),
        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
    )]));

    if let Some(theme) = &job.plan.theme {
        let instrumentation = if theme.instrumentation.is_empty() {
            "blended instrumentation".to_string()
        } else {
            theme.instrumentation.join(", ")
        };
        lines.push(Line::from(vec![
            Span::styled(format!("Motif: {}", theme.motif), Style::default().fg(Color::LightCyan)),
            Span::raw(" | "),
            Span::styled(
                format!("Instruments: {}", instrumentation),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw(" | "),
            Span::styled(
                format!("Rhythm: {}", theme.rhythm),
                Style::default().fg(Color::LightBlue),
            ),
        ]));
    }

    let active_label = job
        .status
        .message
        .as_ref()
        .and_then(|m| m.strip_prefix("rendering "))
        .and_then(|rest| rest.split(':').nth(1))
        .map(|s| s.trim())
        .map(|s| s.split('(').next().unwrap_or(s).trim().to_string());

    let section_extras = extract_section_extras(job);

    for section in &job.plan.sections {
        lines.push(plan_section_line(section, &section_extras, active_label.as_deref()));
    }

    lines
}

struct ConditioningSummary {
    phrase_sections: usize,
    avg_padding: f32,
    avg_tail: f32,
    max_tail: f32,
    missing_tails: usize,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct SectionExtras {
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    placeholder: Option<bool>,
    #[serde(default)]
    render_seconds: Option<f32>,
    #[serde(default)]
    phrase: Option<PhraseExtras>,
    #[serde(default)]
    arrangement_text: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    orchestration: Option<SectionOrchestration>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PhraseExtras {
    #[serde(default)]
    seconds: Option<f32>,
    #[serde(default)]
    beats: Option<f32>,
    #[serde(default)]
    bars: Option<i32>,
    #[serde(default)]
    padding_seconds: Option<f32>,
    #[serde(default)]
    conditioning_tail_seconds: Option<f32>,
}

fn extract_section_extras(job: &JobEntry) -> HashMap<String, SectionExtras> {
    let mut map: HashMap<String, SectionExtras> = HashMap::new();
    if let Some(artifact) = &job.artifact {
        if let Some(extras) = artifact.descriptor.metadata.extras.as_object() {
            if let Some(sections) = extras.get("sections").and_then(|value| value.as_array()) {
                for entry in sections {
                    if let Some(section_id) =
                        entry.get("section_id").and_then(|value| value.as_str())
                    {
                        match serde_json::from_value::<SectionExtras>(entry.clone()) {
                            Ok(parsed) => {
                                map.insert(section_id.to_string(), parsed);
                            }
                            Err(err) => {
                                map.insert(
                                    section_id.to_string(),
                                    SectionExtras {
                                        backend: entry
                                            .get("backend")
                                            .and_then(|value| value.as_str())
                                            .map(|s| s.to_string()),
                                        placeholder: entry
                                            .get("placeholder")
                                            .and_then(|value| value.as_bool()),
                                        render_seconds: entry
                                            .get("render_seconds")
                                            .and_then(|value| value.as_f64())
                                            .map(|v| v as f32),
                                        phrase: None,
                                        arrangement_text: None,
                                        orchestration: None,
                                    },
                                );
                                eprintln!(
                                    "failed to parse section extras for {}: {}",
                                    section_id, err
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    map
}

fn conditioning_summary(job: &JobEntry) -> Option<ConditioningSummary> {
    let extras = extract_section_extras(job);
    if extras.is_empty() {
        return None;
    }

    let mut phrase_sections = 0;
    let mut padding_sum = 0.0;
    let mut padding_count = 0;
    let mut tail_sum = 0.0;
    let mut tail_count = 0;
    let mut max_tail = 0.0;
    let mut missing_tails = 0;

    for extra in extras.values() {
        if let Some(phrase) = &extra.phrase {
            phrase_sections += 1;
            if let Some(padding) = phrase.padding_seconds {
                padding_sum += padding;
                padding_count += 1;
            }
            match phrase.conditioning_tail_seconds {
                Some(tail) if tail > 0.0 => {
                    tail_sum += tail;
                    tail_count += 1;
                    if tail > max_tail {
                        max_tail = tail;
                    }
                }
                _ => missing_tails += 1,
            }
        }
    }

    if phrase_sections == 0 {
        return None;
    }

    let avg_padding = if padding_count > 0 { padding_sum / padding_count as f32 } else { 0.0 };
    let avg_tail = if tail_count > 0 { tail_sum / tail_count as f32 } else { 0.0 };

    Some(ConditioningSummary { phrase_sections, avg_padding, avg_tail, max_tail, missing_tails })
}

fn plan_section_line(
    section: &CompositionSection,
    extras: &HashMap<String, SectionExtras>,
    active_label: Option<&str>,
) -> Line<'static> {
    let is_active =
        active_label.map(|label| label.eq_ignore_ascii_case(&section.label)).unwrap_or(false);

    let energy_label = match section.energy {
        SectionEnergy::Low => "low",
        SectionEnergy::Medium => "medium",
        SectionEnergy::High => "high",
    };

    let mut spans: Vec<Span> = Vec::new();
    spans.push(Span::styled(
        if is_active { "▶" } else { " " },
        if is_active { glow_style() } else { Style::default().fg(Color::DarkGray) },
    ));
    spans.push(Span::raw(" "));
    spans.push(Span::styled(
        format!("{} {}", section.section_id, section.label),
        if is_active { glow_style() } else { Style::default().fg(Color::White) },
    ));
    spans.push(Span::styled(
        format!(" · {} · {} bars · {:.1}s", energy_label, section.bars, section.target_seconds),
        Style::default().fg(Color::Gray),
    ));

    if let Some(extra) = extras.get(&section.section_id) {
        if let Some(ref backend) = extra.backend {
            spans.push(Span::styled(format!(" · {}", backend), Style::default().fg(Color::Cyan)));
        }
        if extra.placeholder.unwrap_or(false) {
            spans.push(Span::styled(" · placeholder", Style::default().fg(Color::Magenta)));
        }
        if let Some(render_seconds) = extra.render_seconds {
            spans.push(Span::styled(
                format!(" · render {:.2}s", render_seconds),
                Style::default().fg(Color::LightGreen),
            ));
        }
        if let Some(arrangement) =
            extra.arrangement_text.as_ref().filter(|text| !text.trim().is_empty())
        {
            spans.push(Span::raw(" · "));
            spans.push(Span::styled(
                truncate_text(arrangement, 80),
                Style::default().fg(Color::LightBlue),
            ));
        }
        if let Some(phrase) = &extra.phrase {
            let seconds = phrase.seconds.unwrap_or(section.target_seconds);
            let beats =
                phrase.beats.map(|b| format!("{b:.1} beats")).unwrap_or_else(|| "- beats".into());
            let bars = phrase.bars.map(|b| format!("{b} bars")).unwrap_or_else(|| "- bars".into());
            let padding = phrase
                .padding_seconds
                .map(|p| format!("pad {:.2}s", p))
                .unwrap_or_else(|| "pad 0.00s".into());
            spans.push(Span::raw(" · "));
            spans.push(Span::styled(
                format!("phrase {:.2}s · {beats} · {bars} · {padding}", seconds),
                Style::default().fg(Color::LightYellow),
            ));
            if let Some(tail) = phrase.conditioning_tail_seconds {
                spans.push(Span::styled(
                    format!(" (tail {:.2}s)", tail),
                    Style::default().fg(Color::LightMagenta),
                ));
            }
        }
    }

    Line::from(spans)
}

fn glow_style() -> Style {
    Style::default().fg(glow_color(1.0)).add_modifier(Modifier::BOLD)
}

fn glow_intensity(last_update: DateTime<Utc>, now: DateTime<Utc>) -> f32 {
    let delta_ms = now.signed_duration_since(last_update).num_milliseconds().max(0);
    let elapsed = delta_ms as f32 / 1000.0;
    (1.0 - (elapsed / 2.0)).clamp(0.0, 1.0)
}

fn glow_color(intensity: f32) -> Color {
    let clamp = |value: f32| value.clamp(0.0, 255.0) as u8;
    let red = clamp(210.0 + 40.0 * intensity);
    let green = clamp(140.0 + 30.0 * intensity);
    let blue = clamp(120.0 + 25.0 * intensity);
    Color::Rgb(red, green, blue)
}

fn format_duration(duration: StdDuration) -> String {
    let total_seconds = duration.as_secs();
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;
    format!("{minutes:02}:{seconds:02}")
}

fn pane_border(app: &AppState, pane: FocusedPane) -> Style {
    if app.focused_pane == pane {
        match app.input_mode {
            InputMode::Insert => insert_border_style(),
            InputMode::Normal => glow_style(),
        }
    } else {
        Style::default()
    }
}

fn insert_border_style() -> Style {
    Style::default().fg(Color::Rgb(70, 70, 80)).add_modifier(Modifier::BOLD)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SectionOrchestration, SectionRole};

    #[test]
    fn format_section_line_includes_backend_and_active_marker() {
        let section = CompositionSection {
            section_id: "s00".to_string(),
            role: SectionRole::Motif,
            label: "Theme".to_string(),
            prompt: "melody".to_string(),
            bars: 4,
            target_seconds: 8.0,
            energy: SectionEnergy::Medium,
            model_id: None,
            seed_offset: Some(0),
            transition: None,
            motif_directive: Some("state motif".to_string()),
            variation_axes: vec!["motif fidelity".to_string()],
            cadence_hint: Some("open cadence".to_string()),
            orchestration: SectionOrchestration::default(),
        };
        let mut extras: HashMap<String, SectionExtras> = HashMap::new();
        extras.insert(
            "s00".to_string(),
            SectionExtras {
                backend: Some("musicgen".to_string()),
                placeholder: Some(false),
                render_seconds: Some(8.4),
                phrase: Some(PhraseExtras {
                    seconds: Some(8.4),
                    beats: Some(16.0),
                    bars: Some(4),
                    padding_seconds: Some(0.35),
                    conditioning_tail_seconds: Some(3.2),
                }),
                arrangement_text: Some("Feature the arrangement with layered guitars.".to_string()),
                orchestration: Some(SectionOrchestration {
                    lead: vec!["layered guitars".to_string()],
                    ..SectionOrchestration::default()
                }),
            },
        );

        let line = plan_section_line(&section, &extras, Some("Theme"));
        let rendered = line.spans.iter().map(|span| span.content.as_ref()).collect::<String>();
        assert!(rendered.contains("musicgen"));
        assert!(rendered.contains("render 8.40s"));
        assert!(rendered.contains("16.0 beats"));
        assert!(rendered.contains("4 bars"));
        assert!(rendered.trim_start().starts_with("▶"));
        assert!(rendered.contains("phrase"));
        assert!(rendered.contains("tail 3.20s"));
    }
}
