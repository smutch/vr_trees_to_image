#![feature(iter_collect_into)]

use ::std::collections::HashSet;
use std::path::PathBuf;

use hdf5::{File, Result};
use log::info;
use ndarray::{s, Array1, Array2, ArrayView1, Axis};


fn read_total_snaps(fin: &File) -> Result<usize> {
    Ok(fin
        .member_names()?
        .iter()
        .filter(|n| n.starts_with("Snap_"))
        .count())
}

#[derive(Debug)]
pub struct Pixel {
    pub snap: usize,
    pub col: usize,
    pub mass: f32,
    pub typ: i8,
    pub displacement: f32,
}

#[derive(Debug)]
pub struct HaloProps {
    progenitors: Vec<Array1<u64>>,
    next_progenitors: Vec<Array1<u64>>,
    masses: Vec<Array1<f32>>,
    types: Vec<Array1<i8>>,
    positions: Vec<Array2<f32>>,
}

#[derive(Debug)]
pub struct ImageProps {
    pub first_snap: usize,
    pub last_snap: usize,
    pub n_rows: usize,
    pub n_cols: usize,
}


#[inline]
fn id_to_snap(id: u64) -> usize {
    (id / 10u64.pow(12)).try_into().unwrap()
}

#[inline]
fn id_to_ind(id: u64) -> usize {
    ((id % 10u64.pow(12)) - 1).try_into().unwrap()
}

impl HaloProps {
    fn new(size: usize) -> Self {
        Self {
            progenitors: Vec::with_capacity(size),
            next_progenitors: Vec::with_capacity(size),
            masses: Vec::with_capacity(size),
            types: Vec::with_capacity(size),
            positions: Vec::with_capacity(size),
        }
    }
}

fn id_to_snap_ind(id: u64) -> (usize, usize) {
    (id_to_snap(id), id_to_ind(id))
}


pub fn reorder_progenitors(id: u64, depth: u32, halo_props: &mut HaloProps) -> u32 {
    let mut max_depth = depth;

    let (snap, ind) = id_to_snap_ind(id);
    let mut prog_id = halo_props.progenitors[snap][ind];
    if prog_id != id {
        max_depth = reorder_progenitors(prog_id, depth + 1, halo_props);
        let mut first_prog_depth = max_depth;

        let mut cur_id = prog_id;
        let (mut prog_snap, mut prog_ind) = id_to_snap_ind(prog_id);
        let mut next_id = halo_props.next_progenitors[prog_snap][prog_ind];

        #[cfg(debug_assertions)]
        let mut np_counter = 0u32;

        while next_id != cur_id && next_id != id {
            max_depth = max_depth.max(reorder_progenitors(next_id, depth + 1, halo_props));
            cur_id = next_id;
            let (next_snap, next_ind) = id_to_snap_ind(next_id);
            next_id = halo_props.next_progenitors[next_snap][next_ind];

            if max_depth > first_prog_depth {
                halo_props.progenitors[snap][ind] = cur_id;
                halo_props.next_progenitors[next_snap][next_ind] = prog_id;
                if next_id != cur_id && next_id != id {
                    halo_props.next_progenitors[prog_snap][prog_ind] = next_id;
                } else {
                    halo_props.next_progenitors[prog_snap][prog_ind] = prog_id;
                }
                prog_id = cur_id;
                (prog_snap, prog_ind) = id_to_snap_ind(prog_id);
                first_prog_depth = max_depth;
            }

            #[cfg(debug_assertions)]
            {
                np_counter += 1;
                if np_counter > 5000 {
                    panic!(
                        "I don't think we should have >5k next progenitors! Something has gone wrong!"
                        );
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    if max_depth > halo_props.progenitors.len().try_into().unwrap() {
        panic!("Somehow we have a max_depth > the number of snapshots! Something has gone wrong!");
    }

    max_depth
}

fn lerp(
    a_id: u64,
    b_id: u64,
    ref_pos: ArrayView1<f32>,
    col: usize,
    halo_props: &HaloProps,
    pixels: &mut Vec<Pixel>,
) {
    let (a_snap, a_ind) = id_to_snap_ind(a_id);
    let (b_snap, b_ind) = id_to_snap_ind(b_id);

    let a_mass = halo_props.masses[a_snap][a_ind];
    let b_mass = halo_props.masses[b_snap][b_ind];

    let typ = halo_props.types[a_snap][a_ind];

    let a_pos = halo_props.positions[a_snap]
        .slice(s![a_ind, ..])
        .into_owned();
    let a_disp = (a_pos - ref_pos).mapv(|v| v.powi(2)).sum().sqrt();
    let b_pos = halo_props.positions[b_snap]
        .slice(s![b_ind, ..])
        .into_owned();
    let b_disp = (b_pos - ref_pos).mapv(|v| v.powi(2)).sum().sqrt();

    for snap in a_snap..b_snap {
        let frac = (snap - a_snap) as f32 / (b_snap - a_snap) as f32;
        let mass = a_mass + (b_mass - a_mass) * frac;
        let disp = a_disp + (b_disp - a_disp) * frac;
        pixels.push(Pixel {
            snap,
            col,
            mass,
            typ,
            displacement: disp,
        });
    }
}

pub fn place_pixels(id: u64, halo_props: &HaloProps) -> (Vec<Pixel>, ImageProps) {
    let mut pixels: Vec<Pixel> = Vec::new();
    let ref_pos = halo_props.positions[id_to_snap(id)].slice(s![id_to_ind(id), ..]);

    walk_and_place_pixels(
        id,
        &mut pixels,
        ref_pos,
        0,
        &mut 0,
        halo_props,
        // id == 619000000000517
    );

    let first_snap = pixels.iter().map(|v| v.snap).min().unwrap();
    let last_snap = pixels.iter().map(|v| v.snap).max().unwrap();
    let n_rows = last_snap - first_snap + 1;
    for v in pixels.iter_mut() {
        v.snap -= first_snap;
    }
    let n_cols = pixels.iter().map(|v| v.col).max().unwrap() + 1;

    let image_props = ImageProps {
        first_snap,
        last_snap,
        n_rows,
        n_cols,
    };

    (pixels, image_props)
}

fn walk_and_place_pixels(
    id: u64,
    pixels: &mut Vec<Pixel>,
    ref_pos: ArrayView1<f32>,
    col: usize,
    max_col: &mut usize,
    halo_props: &HaloProps,
) {
    let (snap, ind) = id_to_snap_ind(id);
    let prog_id = halo_props.progenitors[snap][ind];

    if prog_id != id {
        walk_and_place_pixels(prog_id, pixels, ref_pos, col, max_col, halo_props);

        let mut cur_id = prog_id;
        let (prog_snap, prog_ind) = id_to_snap_ind(prog_id);

        if snap - prog_snap > 1 {
            lerp(id, prog_id, ref_pos, col, halo_props, pixels);
        }

        let mut next_id = halo_props.next_progenitors[prog_snap][prog_ind];
        while next_id != cur_id && next_id != id {
            *max_col += 1;
            walk_and_place_pixels(next_id, pixels, ref_pos, *max_col, max_col, halo_props);

            let (next_snap, next_ind) = id_to_snap_ind(next_id);

            if snap - next_snap > 1 {
                lerp(id, next_id, ref_pos, col, halo_props, pixels);
            }

            cur_id = next_id;
            next_id = halo_props.next_progenitors[next_snap][next_ind];

            #[cfg(debug_assertions)]
            if *max_col > 50000 {
                panic!("I don't think we should have >5k wide tree! Something has gone wrong!");
            }
        }
    }

    let pos = halo_props.positions[snap].slice(s![ind, ..]).into_owned();
    let displacement = (pos - ref_pos).mapv(|v| v.powi(2)).sum().sqrt();

    pixels.push(Pixel {
        snap,
        col,
        mass: halo_props.masses[snap][ind],
        typ: halo_props.types[snap][ind],
        displacement,
    });
}

pub fn read_halos(trees_path: PathBuf) -> Result<HaloProps> {
    let fin = File::open(trees_path)?;
    let total_snaps = read_total_snaps(&fin)?;

    let mut halo_props = HaloProps::new(total_snaps);
    for snap in 0..total_snaps {
        let group = fin.group(format!("Snap_{snap:03}").as_str())?;
        halo_props
            .progenitors
            .push(group.dataset("Progenitor")?.read_1d::<u64>()?);
        halo_props
            .next_progenitors
            .push(group.dataset("NextProgenitor")?.read_1d::<u64>()?);
        halo_props
            .masses
            .push(group.dataset("Mass_200crit")?.read_1d::<f32>()?);
        halo_props.types.push(
            group
                .dataset("hostHaloID")?
                .read_1d::<i64>()?
                .mapv(|v| (v != -1).into()),
        );
        halo_props.positions.push(ndarray::stack![
            Axis(1),
            group.dataset("Xc")?.read_1d::<f32>()?,
            group.dataset("Yc")?.read_1d::<f32>()?,
            group.dataset("Zc")?.read_1d::<f32>()?
        ]);
    }

    Ok(halo_props)
}

pub fn read_final_descendants(trees_path: &PathBuf) -> Result<HashSet<u64>> {
    let fin = File::open(trees_path)?;

    let total_snaps = read_total_snaps(&fin)?;
    let mut final_descendants = HashSet::new();
    for snap in 0..total_snaps {
        fin.dataset(format!("Snap_{snap:03}/FinalDescendant").as_str())?
            .read_1d::<u64>()?
            .iter()
            .collect_into(&mut final_descendants);
    }
    let n_final_descendants = final_descendants.len();
    info!("Found {n_final_descendants} unique FinalDescendant IDs");

    Ok(final_descendants)
}

pub fn dump_final_descendants(
    fname_out: PathBuf,
    final_descendants: HashSet<u64>,
) -> std::result::Result<(), std::io::Error> {
    use std::io::Write;
    let mut fout = std::fs::File::create(fname_out)?;
    for final_desc in final_descendants {
        writeln!(fout, "{}", final_desc)?;
    }
    Ok(())
}

