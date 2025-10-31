mod note;

use crate::app::{AppState, SplashSelection};
use note::RotatingEighthNote;
use once_cell::sync::Lazy;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use std::time::Duration as StdDuration;

const SPLASH_TAGLINE: &str = "A generative CLI tool for the musician";
const SPLASH_INSTRUCTIONS: &str = "Use Up/Down to choose | Enter to continue";

static SPLASH_RENDERER: Lazy<SplashRenderer> = Lazy::new(SplashRenderer::new);

pub fn render(frame: &mut ratatui::Frame, app: &AppState) {
    let context = SplashContext {
        elapsed: app.splash_elapsed(),
        restore_available: app.has_pending_snapshot(),
        selection: app.splash_selection(),
    };
    SPLASH_RENDERER.render(frame, frame.size(), context);
}

struct SplashRenderer {
    note: RotatingEighthNote,
}

struct SplashContext {
    elapsed: StdDuration,
    restore_available: bool,
    selection: SplashSelection,
}

impl SplashRenderer {
    fn new() -> Self {
        Self { note: RotatingEighthNote::default() }
    }

    fn render(&self, frame: &mut ratatui::Frame, area: Rect, context: SplashContext) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(20), Constraint::Length(6), Constraint::Length(4)])
            .split(area);

        self.render_note(frame, chunks[0], context.elapsed);
        self.render_options(frame, chunks[1], &context);

        let footer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(2), Constraint::Length(2)])
            .split(chunks[2]);

        self.render_instructions(frame, footer[0]);
        self.render_tagline(frame, footer[1]);
    }

    fn render_note(&self, frame: &mut ratatui::Frame, area: Rect, elapsed: StdDuration) {
        let block = Block::default().borders(Borders::ALL).title("Timbre AI");
        let inner = block.inner(area);
        let note_lines = self.note.render(inner.width, inner.height, elapsed);
        let padded = self.center_vertically(note_lines, inner.height as usize);
        let paragraph = Paragraph::new(padded).alignment(Alignment::Center).block(block);
        frame.render_widget(paragraph, area);
    }

    fn render_options(&self, frame: &mut ratatui::Frame, area: Rect, context: &SplashContext) {
        let option_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        self.render_option(
            frame,
            option_chunks[0],
            "Restore Session",
            if context.restore_available {
                "Load the most recent snapshot and continue working."
            } else {
                "No snapshot available yet — generate something to save one."
            },
            context.selection == SplashSelection::Restore,
            context.restore_available,
        );

        self.render_option(
            frame,
            option_chunks[1],
            "New Session",
            "Start a fresh composition with a blank timeline.",
            context.selection == SplashSelection::StartNew,
            true,
        );
    }

    fn render_option(
        &self,
        frame: &mut ratatui::Frame,
        area: Rect,
        title: &str,
        description: &str,
        selected: bool,
        enabled: bool,
    ) {
        let mut title_style =
            Style::default().fg(if enabled { Color::White } else { Color::DarkGray });
        if selected && enabled {
            title_style = title_style.fg(Color::Yellow).add_modifier(Modifier::BOLD);
        } else if selected {
            title_style = title_style.add_modifier(Modifier::BOLD);
        }

        let border_color = if selected { Color::Yellow } else { Color::DarkGray };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .title(Span::styled(title, title_style));

        let text_style = if enabled {
            if selected {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            }
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let lines =
            vec![Line::from(""), Line::from(Span::styled(description, text_style)), Line::from("")];

        let paragraph = Paragraph::new(lines)
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true })
            .block(block);

        frame.render_widget(paragraph, area);
    }

    fn render_instructions(&self, frame: &mut ratatui::Frame, area: Rect) {
        let instructions = Paragraph::new(SPLASH_INSTRUCTIONS)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM));
        frame.render_widget(instructions, area);
    }

    fn render_tagline(&self, frame: &mut ratatui::Frame, area: Rect) {
        let tagline = Paragraph::new(SPLASH_TAGLINE)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC));
        frame.render_widget(tagline, area);
    }

    fn center_vertically(
        &self,
        lines: Vec<Line<'static>>,
        target_height: usize,
    ) -> Vec<Line<'static>> {
        if target_height == 0 {
            return Vec::new();
        }

        let line_count = lines.len();
        if line_count >= target_height {
            return lines;
        }

        let padding = target_height - line_count;
        let top_padding = padding / 2;
        let bottom_padding = padding - top_padding;

        let mut output = Vec::with_capacity(target_height);
        for _ in 0..top_padding {
            output.push(Line::from(""));
        }
        output.extend(lines);
        for _ in 0..bottom_padding {
            output.push(Line::from(""));
        }
        output
    }
}
