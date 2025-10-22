use anyhow::Result;
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;
use tracing::info;

mod api;
mod ui;
mod types;

#[tokio::main]
async fn main() -> Result<()> {
    setup_tracing()?;
    info!("starting timbre CLI prototype");

    let base_url = std::env::var("TIMBRE_WORKER_URL").ok();
    let client = api::Client::new(base_url.as_deref())?;

    let mut app = ui::App::new();
    match client.health_check().await {
        Ok(_) => app
            .status_lines
            .push(format!("Worker health: ok ({})", client.base_url())),
        Err(err) => app
            .status_lines
            .push(format!("Worker health: error ({err})")),
    }

    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    ui::run(&mut terminal, &mut app)?;

    Ok(())
}

fn setup_tracing() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .compact()
        .try_init()
        .map_err(|err| anyhow::anyhow!(err))?;
    Ok(())
}
