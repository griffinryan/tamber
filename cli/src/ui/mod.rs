use crate::{
    app::{format_request_summary, AppCommand, AppEvent, AppState, ChatRole, JobEntry},
    types::{CompositionSection, JobState, SectionEnergy, SectionRole},
};
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Terminal,
};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::mpsc::{error::TryRecvError, UnboundedReceiver, UnboundedSender};

pub fn run<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut AppState,
    event_rx: &mut UnboundedReceiver<AppEvent>,
    command_tx: UnboundedSender<AppCommand>,
) -> Result<()> {
    enable_raw_mode()?;
    let result = run_loop(terminal, app, event_rx, command_tx);
    disable_raw_mode()?;
    result
}

fn run_loop<B: ratatui::backend::Backend>(
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

    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(3)])
        .split(layout[0]);

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
    let mut lines: Vec<Line> = app
        .chat
        .iter()
        .rev()
        .take(200)
        .map(|entry| {
            let prefix = format!("[{}]", entry.role.label());
            let ts = entry.timestamp.format("%H:%M:%S").to_string();
            Line::from(vec![
                Span::styled(ts, Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM)),
                Span::raw(" "),
                Span::styled(prefix, chat_style(&entry.role)),
                Span::raw(" "),
                Span::raw(entry.content.clone()),
            ])
        })
        .collect();
    lines.reverse();

    let block = Block::default().title("Conversation").borders(Borders::ALL);
    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}

fn render_prompt(frame: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let block = Block::default()
        .title("Prompt")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    let paragraph = Paragraph::new(app.input.clone()).block(block);
    frame.render_widget(paragraph, area);
}

fn render_jobs(frame: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let items: Vec<ListItem> =
        app.jobs_iter().rev().map(|(job_id, job)| job_list_item(app, job_id, job)).collect();

    let list = List::new(items)
        .block(Block::default().title("Jobs").borders(Borders::ALL))
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
    let mut lines = vec![format!("Config: {}", app.config_summary())];
    if app.status_lines.is_empty() {
        lines.push(String::from("No activity yet."));
    } else {
        lines.extend(app.status_lines.iter().cloned());
    }

    if let Some((_id, job)) = app.selected_job() {
        lines.push(String::new());
        lines.extend(plan_status_lines(job));
    }

    let content = lines.join("\n");
    let block = Block::default().title("Status").borders(Borders::ALL);
    frame.render_widget(Paragraph::new(content).block(block), area);
}

fn handle_key(
    app: &mut AppState,
    command_tx: &UnboundedSender<AppCommand>,
    key: crossterm::event::KeyEvent,
) -> Result<bool> {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(true),
        KeyCode::Char('q') => return Ok(true),
        KeyCode::Char('p') => {
            if let Some((job_id, job)) = app.selected_job() {
                if job.artifact.is_some() {
                    let _ = command_tx.send(AppCommand::PlayJob { job_id: job_id.clone() });
                } else {
                    app.push_status_line(format!("Job {} has no artifact yet", job_id));
                }
            }
        }
        KeyCode::Up => app.select_previous_job(),
        KeyCode::Down => app.select_next_job(),
        KeyCode::Backspace => {
            app.input.pop();
        }
        KeyCode::Esc => {
            app.input.clear();
        }
        KeyCode::Enter => {
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
        }
        KeyCode::Char(c) => {
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

fn plan_status_lines(job: &JobEntry) -> Vec<String> {
    let mut lines = Vec::new();
    let header = format!(
        "Plan: {} sections · {} BPM · {}",
        job.plan.sections.len(),
        job.plan.tempo_bpm,
        job.plan.key
    );
    lines.push(header);

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
        lines.push(format_section_line(section, &section_extras, active_label.as_deref()));
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

fn format_section_line(
    section: &CompositionSection,
    extras: &HashMap<String, serde_json::Value>,
    active_label: Option<&str>,
) -> String {
    let mut label = String::new();
    if let Some(active) = active_label {
        if active.eq_ignore_ascii_case(&section.label) {
            label.push('▶');
            label.push(' ');
        }
    }
    let energy_label = match section.energy {
        SectionEnergy::Low => "low",
        SectionEnergy::Medium => "medium",
        SectionEnergy::High => "high",
    };
    label.push_str(&format!(
        "{} {} · {} · {} bars · {:.1}s",
        section.section_id, section.label, energy_label, section.bars, section.target_seconds,
    ));

    if let Some(extra) = extras.get(&section.section_id) {
        if let Some(backend) = extra.get("backend").and_then(|value| value.as_str()) {
            label.push_str(&format!(" · {}", backend));
        }
        if extra.get("placeholder").and_then(|value| value.as_bool()).unwrap_or(false) {
            label.push_str(" · placeholder");
        }
    }

    label
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let line = format_section_line(&section, &extras, Some("Theme"));
        assert!(line.contains("musicgen"));
        assert!(line.starts_with("▶"));
    }
}
