use crate::{
    app::{
        format_request_summary, AppCommand, AppEvent, AppState, ChatRole, FocusedPane, InputMode,
        JobEntry,
    },
    types::{CompositionSection, JobState, SectionEnergy},
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Terminal,
};
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

    render_chat(frame, left[0], app);
    render_prompt(frame, left[1], app);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(layout[1]);

    render_jobs(frame, right[0], app);
    render_status(frame, right[1], app);
}

fn render_chat(frame: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let mut lines: Vec<Line> = Vec::new();

    for entry in &app.chat {
        let prefix = format!("[{}]", entry.role.label());
        let ts = entry.timestamp.format("%H:%M:%S").to_string();
        lines.push(Line::from(vec![
            Span::styled(ts, Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM)),
            Span::raw(" "),
            Span::styled(prefix, chat_style(&entry.role)),
        ]));

        for line in entry.content.lines() {
            lines.push(Line::from(vec![Span::styled(
                format!("  {line}"),
                Style::default().fg(Color::White),
            )]));
        }

        lines.push(Line::from(Span::raw("")));
    }

    if lines.is_empty() {
        lines.push(Line::from("No messages yet."));
    }

    let mut block = Block::default().title("Conversation").borders(Borders::ALL);
    if app.focused_pane == FocusedPane::Conversation {
        block = block.border_style(pane_border(app, FocusedPane::Conversation));
    }

    let total_lines = lines.len();
    let height = area.height as usize;
    let max_offset = total_lines.saturating_sub(height);
    let desired = if app.chat_following {
        max_offset
    } else {
        max_offset.saturating_sub(app.chat_scroll.min(max_offset))
    };
    let paragraph =
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }).scroll((desired as u16, 0));
    frame.render_widget(paragraph, area);
}

fn render_prompt(frame: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let mut block = Block::default().title("Prompt").borders(Borders::ALL);
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
        KeyCode::Char('q') => return Ok(true),
        KeyCode::Char('s') if matches!(app.focused_pane, FocusedPane::Status) => {
            let _ = command_tx.send(AppCommand::StopPlayback);
        }
        KeyCode::Char('i') => {
            if app.input_mode != InputMode::Insert {
                app.input_mode = InputMode::Insert;
                if matches!(app.focused_pane, FocusedPane::Conversation) {
                    app.chat_following = false;
                }
                if matches!(app.focused_pane, FocusedPane::Status) {
                    app.status_following = false;
                }
            }
        }
        KeyCode::Esc => {
            if app.input_mode == InputMode::Insert {
                app.input_mode = InputMode::Normal;
            }
            if matches!(app.focused_pane, FocusedPane::Conversation) {
                app.chat_scroll = 0;
                app.chat_following = true;
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
                    FocusedPane::Conversation => app.increment_chat_scroll(1),
                    FocusedPane::Status => app.increment_status_scroll(1),
                    FocusedPane::Jobs => app.select_previous_job(),
                    _ => {}
                }
            } else {
                match app.focused_pane {
                    FocusedPane::Prompt => app.focused_pane = FocusedPane::Status,
                    FocusedPane::Conversation => app.increment_chat_scroll(1),
                    FocusedPane::Status => app.increment_status_scroll(1),
                    FocusedPane::Jobs => app.select_previous_job(),
                    _ => {}
                }
            }
        }
        KeyCode::Down => {
            if matches!(app.input_mode, InputMode::Insert) {
                match app.focused_pane {
                    FocusedPane::Conversation => app.increment_chat_scroll(-1),
                    FocusedPane::Status => app.increment_status_scroll(-1),
                    FocusedPane::Jobs => app.select_next_job(),
                    _ => {}
                }
            } else {
                match app.focused_pane {
                    FocusedPane::StatusBar => app.focused_pane = FocusedPane::Conversation,
                    FocusedPane::Conversation => app.increment_chat_scroll(-1),
                    FocusedPane::Status => {
                        if app.status_scroll == 0 {
                            app.focused_pane = FocusedPane::Prompt;
                        } else {
                            app.increment_status_scroll(-1);
                        }
                    }
                    FocusedPane::Jobs => app.select_next_job(),
                    _ => {}
                }
            }
        }
        KeyCode::Left if matches!(app.input_mode, InputMode::Normal) => app.focus_previous(),
        KeyCode::Right if matches!(app.input_mode, InputMode::Normal) => app.focus_next(),
        KeyCode::Tab if matches!(app.input_mode, InputMode::Normal) => app.focus_next(),
        KeyCode::BackTab if matches!(app.input_mode, InputMode::Normal) => app.focus_previous(),
        KeyCode::Backspace
            if matches!(app.focused_pane, FocusedPane::Prompt)
                && matches!(app.input_mode, InputMode::Insert) =>
        {
            app.input.pop();
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

            if line.starts_with('/') {
                match app.handle_command(&line) {
                    Ok(message) => {
                        app.push_status_line(message.clone());
                        app.append_chat(ChatRole::System, message);
                    }
                    Err(err) => {
                        let message = format!("Command error: {err}");
                        app.push_status_line(message.clone());
                        app.append_chat(ChatRole::System, message);
                    }
                }
                app.input.clear();
                return Ok(false);
            }

            let prompt = line;
            let (request, plan) = app.build_generation_payload(&prompt);
            app.append_chat(ChatRole::User, prompt.clone());
            if command_tx.send(AppCommand::SubmitPrompt { prompt, request, plan }).is_err() {
                app.push_status_line("Failed to submit prompt; worker channel closed".into());
            }
            app.input.clear();
            app.focused_pane = FocusedPane::StatusBar;
            app.status_scroll = 0;
            app.status_following = true;
            app.input_mode = InputMode::Normal;
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

fn chat_style(role: &ChatRole) -> Style {
    match role {
        ChatRole::User => Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ChatRole::Worker => Style::default().fg(Color::Green),
        ChatRole::System => Style::default().fg(Color::Yellow),
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

fn extract_section_extras(job: &JobEntry) -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::new();
    if let Some(artifact) = &job.artifact {
        if let Some(extras) = artifact.descriptor.metadata.extras.as_object() {
            if let Some(sections) = extras.get("sections").and_then(|value| value.as_array()) {
                for entry in sections {
                    if let Some(section_id) =
                        entry.get("section_id").and_then(|value| value.as_str())
                    {
                        map.insert(section_id.to_string(), entry.clone());
                    }
                }
            }
        }
    }
    map
}

fn plan_section_line(
    section: &CompositionSection,
    extras: &HashMap<String, serde_json::Value>,
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
        if let Some(backend) = extra.get("backend").and_then(|value| value.as_str()) {
            spans.push(Span::styled(format!(" · {}", backend), Style::default().fg(Color::Cyan)));
        }
        if extra.get("continuation").is_some() {
            spans.push(Span::styled(" · ↺", Style::default().fg(Color::LightMagenta)));
        }
        if extra.get("placeholder").and_then(|value| value.as_bool()).unwrap_or(false) {
            spans.push(Span::styled(" · placeholder", Style::default().fg(Color::Magenta)));
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
    use crate::types::SectionRole;
    use serde_json::json;

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
        };
        let mut extras = HashMap::new();
        extras.insert(
            "s00".to_string(),
            json!({
                "backend": "musicgen",
                "section_id": "s00",
                "placeholder": false
            }),
        );

        let line = plan_section_line(&section, &extras, Some("Theme"));
        let rendered = line.spans.iter().map(|span| span.content.as_ref()).collect::<String>();
        assert!(rendered.contains("musicgen"));
        assert!(rendered.trim_start().starts_with("▶"));
    }
}
