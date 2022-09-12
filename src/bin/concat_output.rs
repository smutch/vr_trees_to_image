use clap::Parser;
use glob::glob;
use hdf5::{File, Result};
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    glob: String,
    out: PathBuf,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Cli::parse();

    let mut fnames: Vec<_> = glob(args.glob.as_str())
        .expect("Failed to parse glob pattern")
        .map(|p| p.expect("Glob result resolution failed"))
        .collect();
    fnames.sort();

    let fout = File::create(args.out)?;

    for fname in fnames {
        let fin = File::open(&fname)?;
        log::info!("Linking {}", fname.display());
        for grp in fin.groups()? {
            fout.link_external(
                fname
                    .to_str()
                    .expect("{fname} does not parse to valid unicode"),
                grp.name().as_str(),
                grp.name().as_str(),
            )?;
        }
    }

    Ok(())
}
