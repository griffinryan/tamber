use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use std::time::Duration;

#[derive(Default)]
pub struct App {
    pub prompt: String,
    pub status_lines: Vec<String>,
    pub history: Vec<String>,
}

impl App {
    pub fn new() -> Self {
        Self::default()
    }
}

pub fn run<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()> {
    enable_raw_mode()?;
    let result = run_loop(terminal, app);
    disable_raw_mode()?;
    result
}

fn run_loop<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(70), Constraint::Percentage(30)].as_ref())
                .split(frame.size());

            let main_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(
                    [
                        Constraint::Min(5),
                        Constraint::Length(3),
                    ]
                    .as_ref(),
                )
                .split(chunks[0]);

            let history_block = Block::default()
                .title("Conversation")
                .borders(Borders::ALL);

            let history_text = app
                .history
                .iter()
                .rev()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");

            frame.render_widget(Paragraph::new(history_text).block(history_block), main_chunks[0]);

            let prompt_block = Block::default()
                .title("Prompt")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
            frame.render_widget(Paragraph::new(app.prompt.as_str()).block(prompt_block), main_chunks[1]);

            let status_block = Block::default().title("Status").borders(Borders::ALL);
            let status_text = Paragraph::new(
                if app.status_lines.is_empty() {
                    "No jobs yet.".into()
                } else {
                    app.status_lines.join("\n")
                },
            )
            .block(status_block);
            frame.render_widget(status_text, chunks[1]);
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char(c) => app.prompt.push(c),
                    KeyCode::Backspace => {
                        app.prompt.pop();
                    }
                    KeyCode::Enter => {
                        if !app.prompt.trim().is_empty() {
                            app.history.push(format!("> {}", app.prompt.trim()));
                            app.status_lines.push("Queued job (stub)".into());
                            app.prompt.clear();
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}
