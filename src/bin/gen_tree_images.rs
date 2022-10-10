#![feature(iter_collect_into)]

use ::std::collections::HashSet;
use std::path::PathBuf;

use clap::Parser;
use hdf5::{File, Result};
use log::info;
use ndarray::Array2;

fn write_unit(name: &str, unit: &str, group: &hdf5::Group) -> Result<()> {
    let attr = group.new_attr::<hdf5::types::VarLenAscii>().create(name)?;
    attr.write_scalar(&hdf5::types::VarLenAscii::from_ascii(unit).unwrap())?;
    Ok(())
}

#[derive(Parser)]
struct Cli {
    #[clap(parse(from_os_str))]
    trees_path: PathBuf,
    #[clap(parse(from_os_str))]
    output_path: PathBuf,
    #[clap(short, long, action)]
    dump_final_descendants: bool,
    #[clap(short, long, parse(from_os_str))]
    read_final_descendants: Option<PathBuf>,
}

fn conditional_pbar<T: ExactSizeIterator>(iter: T) -> indicatif::ProgressBarIter<T> {
    if !log::log_enabled!(log::Level::Debug) {
        indicatif::ProgressIterator::progress(iter)
    } else {
        indicatif::ProgressIterator::progress_with(iter, indicatif::ProgressBar::hidden())
    }
}

fn read_target_ids(fname_in: PathBuf) -> std::result::Result<HashSet<u64>, std::io::Error> {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(&fname_in)?;
    let buffered = BufReader::new(file);

    let mut result = HashSet::new();
    for line in buffered.lines() {
        result.insert(line?.parse::<u64>().unwrap());
    }

    let n_target_ids = result.len();
    info!(
        "Read {n_target_ids} final final_descendant IDs from {:?}",
        fname_in
    );

    Ok(result)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Cli::parse();

    if args.dump_final_descendants {
        let final_descendants = gen_tree_images::read_final_descendants(&args.trees_path)?;
        gen_tree_images::dump_final_descendants(args.output_path, final_descendants).unwrap();
        return Ok(());
    }

    let final_descendants = args
        .read_final_descendants
        .map(|fname| read_target_ids(fname).unwrap())
        .unwrap_or_else(|| gen_tree_images::read_final_descendants(&args.trees_path).unwrap());

    let mut halo_props = gen_tree_images::read_halos(args.trees_path)?;

    let fout = File::create(args.output_path)?;

    for id in conditional_pbar(final_descendants.into_iter()) {
        log::debug!("id = {id}");

        gen_tree_images::reorder_progenitors(id, 0, &mut halo_props);
        let (pixels, image_props) = gen_tree_images::place_pixels(id, &halo_props);

        let group = fout.create_group(format!("{id}").as_str())?;
        group
            .new_attr::<u32>()
            .create("first_snap")?
            .write_scalar(&u32::try_from(image_props.first_snap).unwrap())?;
        group
            .new_attr::<u32>()
            .create("last_snap")?
            .write_scalar(&u32::try_from(image_props.last_snap).unwrap())?;

        macro_rules! construct_and_write {
            ( $prop:ident, $name:literal, $image:ident, $fillval:literal) => {
                $image.fill(num_traits::cast($fillval).unwrap());
                for pixel in pixels.iter() {
                    $image[[pixel.snap, pixel.col]] = pixel.$prop;
                }
                group
                    .new_dataset_builder()
                    // .deflate(7)
                    .with_data(&$image)
                    .create($name)?;
            };
        }

        log::debug!(
            "nrows,ncols = {},{}",
            image_props.n_rows,
            image_props.n_cols
        );

        {
            let mut image: Array2<f32> = Array2::zeros((image_props.n_rows, image_props.n_cols));
            construct_and_write!(mass, "mass", image, 0);
            construct_and_write!(displacement, "displacement", image, 0);
        }

        let mut image: Array2<i8> = Array2::zeros((image_props.n_rows, image_props.n_cols));
        construct_and_write!(typ, "type", image, -1);
    }

    let group = fout.create_group("units")?;

    write_unit("mass", "1e10 Msun", &group)?;
    write_unit("type", "unitless", &group)?;
    write_unit("displacement", "Mpc", &group)?;

    Ok(())
}
