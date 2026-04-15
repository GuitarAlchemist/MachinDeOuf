//! Friday Brief CLI entry point. Single subcommand `run` (default).

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "ix-friday-brief", version, about = "Friday Brief MVP runner")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the Friday Brief pipeline over the synthetic fixture.
    Run {
        /// Which week to brief. Currently ignored; reserved for phase 2
        /// where it will select `last` vs `current` from the real
        /// claude-mem export feed.
        #[arg(long, default_value = "last")]
        week: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let cmd = cli.cmd.unwrap_or(Cmd::Run {
        week: "last".into(),
    });

    match cmd {
        Cmd::Run { week } => {
            if week != "last" {
                eprintln!("note: --week {week} ignored in MVP (phase 2 will honor it)");
            }
            match ix_friday_brief::run() {
                Ok(artifacts) => {
                    println!("{}", ix_friday_brief::describe_artifacts(&artifacts));
                }
                Err(e) => {
                    eprintln!("ix-friday-brief failed: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}
