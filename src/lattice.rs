use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};
use num_traits::float::Float;
use rand::Rng;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;
use std::ops::{Index, IndexMut};

#[derive(Debug)]
pub struct Lattice {
    // x shape same, y cut in half
    a_lattice: Array2<i32>,
    b_lattice: Array2<i32>,
    boundaries: (i32, i32),
    potential: Array1<f64>,
    mc_trials: [i32; 3],
}

impl Index<&(usize, usize)> for Lattice {
    type Output = i32;

    fn index(&self, index: &(usize, usize)) -> &Self::Output {
        let (x, y) = *index;
        match Self::global_to_sub(x, y) {
            SubLatticeIndex::A(x, y) => self.a_lattice.get((x, y)).expect("Bad index"),
            SubLatticeIndex::B(x, y) => self.b_lattice.get((x, y)).expect("Bad index"),
        }
    }
}

impl IndexMut<&(usize, usize)> for Lattice {
    fn index_mut(&mut self, index: &(usize, usize)) -> &mut Self::Output {
        let (x, y) = *index;
        match Self::global_to_sub(x, y) {
            SubLatticeIndex::A(x, y) => self.a_lattice.get_mut((x, y)).expect("Bad index"),
            SubLatticeIndex::B(x, y) => self.b_lattice.get_mut((x, y)).expect("Bad index"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SubLatticeIndex {
    A(usize, usize),
    B(usize, usize),
}

impl Default for SubLatticeIndex {
    fn default() -> Self {
        Self::A(0, 0)
    }
}

impl Lattice {
    pub fn new<V: Into<Array1<f64>>>(
        x: usize,
        y: usize,
        potential: V,
        boundary: (i32, i32),
    ) -> Self {
        assert_eq!(x % 2, 0);
        assert_eq!(y % 2, 0);
        Self {
            a_lattice: Array2::zeros((x, y / 2)),
            b_lattice: Array2::zeros((x, y / 2)),
            boundaries: boundary,
            potential: potential.into(),
            mc_trials: [-1, 0, 1],
        }
    }

    pub fn get_energy(&self, x: usize, y: usize) -> f64 {
        let [lx, ly] = *self.a_lattice.shape() else {unreachable!()};
        let ly = ly * 2;
        let (wx, wy) = self.boundaries;
        let val = self[&(x, y)];

        let neighbors = Self::neighbors(x, y, lx, ly);
        println!("{:?}", neighbors);
        neighbors
            .into_iter()
            .map(|(n, (wrap_x, wrap_y))| {
                let winding_num = (wrap_x as i32 * wx) + (wrap_y as i32 * wy);
                print!("{:?}\t", n);
                let nv = match n {
                    SubLatticeIndex::B(bx, by) => {
                        *self.b_lattice.get((bx, by)).expect("neighbor not found")
                    }
                    SubLatticeIndex::A(ax, ay) => {
                        *self.a_lattice.get((ax, ay)).expect("neighbor not found")
                    }
                };
                nv + winding_num
            })
            .map(|nv| {
                let pot = self.potential.get((nv - val).abs() as usize).copied();
                pot.unwrap_or(f64::infinity())
            })
            .sum::<f64>()
    }

    #[inline]
    fn a_to_global(x: usize, y: usize) -> (usize, usize) {
        (x, 2 * y + (x % 2))
    }

    #[inline]
    fn b_to_global(x: usize, y: usize) -> (usize, usize) {
        (x, 2 * y + ((x + 1) % 2))
    }

    #[inline]
    fn global_to_sub(x: usize, y: usize) -> SubLatticeIndex {
        if (x + y) % 2 == 0 {
            let y = (y - (x % 2)) / 2;
            SubLatticeIndex::A(x, y)
        } else {
            let y = (y - ((x + 1) % 2)) / 2;
            SubLatticeIndex::B(x, y)
        }
    }

    #[inline]
    fn neighbors(x: usize, y: usize, lx: usize, ly: usize) -> [(SubLatticeIndex, (i8, i8)); 4] {
        let x = x as i32;
        let y = y as i32;
        let lx = lx as i32;
        let ly = ly as i32;

        let mut out: [_; 4] = Default::default();
        let deltas = [-1, 1];
        out.iter_mut()
            .zip(
                deltas
                    .iter()
                    .map(|x| (*x, 0))
                    .chain(deltas.iter().map(|x| (0, *x)))
                    .map(|(dx, dy)| {
                        let x_wrap = if (x + dx) < 0 {
                            -1
                        } else if (x + dx) >= lx {
                            1
                        } else {
                            0
                        };
                        let y_wrap = if (y + dy) < 0 {
                            -1
                        } else if (y + dy) >= ly {
                            1
                        } else {
                            0
                        };

                        let nx = ((x + lx) + dx) % lx;
                        let ny = ((y + ly) + dy) % ly;
                        (
                            Self::global_to_sub(nx as usize, ny as usize),
                            (x_wrap, y_wrap),
                        )
                    }),
            )
            .for_each(|(v, x)| {
                *v = x;
            });
        out
    }

    #[inline]
    fn positive_neighbors(
        x: usize,
        y: usize,
        lx: usize,
        ly: usize,
    ) -> [(SubLatticeIndex, (bool, bool)); 2] {
        let x = x as i32;
        let y = y as i32;
        let lx = lx as i32;
        let ly = ly as i32;

        let mut out: [_; 2] = Default::default();
        let deltas = [(0, 1), (1, 0)];
        out.iter_mut()
            .zip(deltas.iter().map(|(dx, dy)| {
                let x_wrap = (x + dx) >= lx;
                let y_wrap = (y + dy) >= ly;

                let nx = (x + dx) % lx;
                let ny = (y + dy) % ly;
                (
                    Self::global_to_sub(nx as usize, ny as usize),
                    (x_wrap, y_wrap),
                )
            }))
            .for_each(|(v, x)| {
                *v = x;
            });
        out
    }

    #[inline]
    fn a_neighbors(ax: usize, ay: usize, lx: usize, ly: usize) -> [((usize, usize), (i8, i8)); 4] {
        let aly = ly / 2;
        let col_parity = ax % 2;

        // -1 x
        let mx = ((ax + lx - 1) % lx, ay);
        let wmx = (if ax == 0 { -1 } else { 0 }, 0);
        // +1 x
        let px = ((ax + lx + 1) % lx, ay);
        let wpx = (if ax + 1 == lx { 1 } else { 0 }, 0);

        if col_parity == 0 {
            // -1 y
            let my = (ax, (ay + aly - 1) % aly);
            let wmy = (0, if ay == 0 { -1 } else { 0 });
            // +1 y (can't go off top on a 0 parity column)
            let py = (ax, ay);
            let wpy = (0, 0);

            [(mx, wmx), (px, wpx), (my, wmy), (py, wpy)]
        } else {
            // -1 y (can't go off bottom on a 1 parity column)
            let my = (ax, ay);
            let wmy = (0, 0);
            // +1 y
            let py = (ax, (ay + aly + 1) % aly);
            let wpy = (0, if ay + 1 == aly { 1 } else { 0 });
            [(mx, wmx), (px, wpx), (my, wmy), (py, wpy)]
        }
    }

    #[inline]
    fn b_neighbors(bx: usize, by: usize, lx: usize, ly: usize) -> [((usize, usize), (i8, i8)); 4] {
        let bly = ly / 2;
        let col_parity = bx % 2;

        // -1 x
        let mx = ((bx + lx - 1) % lx, by);
        let wmx = (if bx == 0 { -1 } else { 0 }, 0);
        // +1 x
        let px = ((bx + lx + 1) % lx, by);
        let wpx = (if bx + 1 == lx { 1 } else { 0 }, 0);

        if col_parity == 0 {
            // -1 y (can't go off bottom on a 1 parity column)
            let my = (bx, by);
            let wmy = (0, 0);
            // +1 y
            let py = (bx, (by + bly + 1) % bly);
            let wpy = (0, if by + 1 == bly { 1 } else { 0 });
            [(mx, wmx), (px, wpx), (my, wmy), (py, wpy)]
        } else {
            // -1 y
            let my = (bx, (by + bly - 1) % bly);
            let wmy = (0, if by == 0 { -1 } else { 0 });
            // +1 y (can't go off top on a 0 parity column)
            let py = (bx, by);
            let wpy = (0, 0);

            [(mx, wmx), (px, wpx), (my, wmy), (py, wpy)]
        }
    }

    pub fn new_update<R: Rng>(&mut self, rng: &mut R) {
        self.new_update_a(rng);
        self.new_update_b(rng);
    }

    fn new_update_a<R: Rng>(&mut self, rng: &mut R) {
        let alpha_lat = self.a_lattice.view_mut();
        let beta_lat = self.b_lattice.view();
        let pots = self.potential.view();
        let winding = self.boundaries;
        let mc_trials = &self.mc_trials;

        Self::update_gen(
            alpha_lat,
            beta_lat,
            pots,
            winding,
            mc_trials,
            Self::a_neighbors,
            rng,
        )
    }

    fn new_update_b<R: Rng>(&mut self, rng: &mut R) {
        let alpha_lat = self.b_lattice.view_mut();
        let beta_lat = self.a_lattice.view();
        let pots = self.potential.view();
        let winding = self.boundaries;
        let mc_trials = &self.mc_trials;

        Self::update_gen(
            alpha_lat,
            beta_lat,
            pots,
            winding,
            mc_trials,
            Self::b_neighbors,
            rng,
        )
    }

    pub fn update_gen<FN, R: Rng, const N: usize>(
        mut alpha_lat: ArrayViewMut2<i32>,
        beta_lat: ArrayView2<i32>,
        pots: ArrayView1<f64>,
        winding: (i32, i32),
        mc_trials: &[i32; N],
        neighbors: FN,
        rng: &mut R,
    ) where
        FN: Fn(usize, usize, usize, usize) -> [((usize, usize), (i8, i8)); 4] + Send + Sync,
    {
        let (wx, wy) = winding;
        let lx = alpha_lat.shape()[0];
        let ly = 2 * alpha_lat.shape()[1];
        ndarray::Zip::indexed(&mut alpha_lat).for_each(|(alpha_x, alpha_y), val| {
            let beta_neighbors = neighbors(alpha_x, alpha_y, lx, ly);

            let mut neighbor_diffs = [0; 4];
            neighbor_diffs
                .iter_mut()
                .zip(beta_neighbors.into_iter())
                .for_each(|(v, ((beta_x, beta_y), (wrap_x, wrap_y)))| {
                    let beta_v = beta_lat
                        .get((beta_x, beta_y))
                        .copied()
                        .expect("Index not found");
                    let winding_num = (wrap_x as i32 * wx) + (wrap_y as i32 * wy);
                    *v = *val - beta_v + winding_num;
                });
            let mut weights = [0.0; N];
            weights
                .iter_mut()
                .zip(mc_trials.into_iter())
                .for_each(|(w, dv)| {
                    *w = neighbor_diffs
                        .iter()
                        .map(|nv| nv + dv)
                        .map(|v| {
                            pots.get([v.abs() as usize])
                                .copied()
                                .unwrap_or(f64::INFINITY)
                        })
                        .sum::<f64>();
                });
            let min_e = weights
                .iter()
                .copied()
                .fold(f64::infinity(), |acc, x| acc.min(x));
            weights.iter_mut().for_each(|x| *x = (-(*x - min_e)).exp());

            // Probs is now the set of boltzman weights for each of the dvs
            let total_weight = weights.iter().sum::<f64>();
            // Probs is now the acceptance probabilities for each of the dvs

            let choice = rng.gen::<f64>() * total_weight;
            let dv = weights
                .into_iter()
                .zip(mc_trials.iter().copied())
                .try_fold(choice, |mut acc, (w, dv)| {
                    acc -= w;
                    if acc <= 0.0 {
                        Err(dv)
                    } else {
                        Ok(acc)
                    }
                })
                .unwrap_err();
            *val += dv;
        });
    }

    pub fn old_update(&mut self) {
        self.old_update_a();
        self.old_update_b();
    }

    fn old_update_a(&mut self) {
        let (wx, wy) = self.boundaries;
        let lx = self.a_lattice.shape()[0];
        let ly = 2 * self.a_lattice.shape()[1];
        let b_sub = self.b_lattice.view();
        let pots = self.potential.view();
        let dvs = &self.mc_trials;
        ndarray::Zip::indexed(&mut self.a_lattice)
            .into_par_iter()
            .for_each(|((ix, iy), val)| {
                let v = *val;
                let (x, y) = Self::a_to_global(ix, iy);
                let mut neighbors = [0; 4];
                neighbors
                    .iter_mut()
                    .zip(Self::neighbors(x, y, lx, ly).into_iter())
                    .for_each(|(v, (n, (wrap_x, wrap_y)))| {
                        let (bx, by) = match n {
                            SubLatticeIndex::B(x, y) => (x, y),
                            SubLatticeIndex::A(_, _) => unreachable!(),
                        };
                        let winding_num = (wrap_x as i32 * wx) + (wrap_y as i32 * wy);
                        *v = *b_sub.get((bx, by)).expect("neighbor not found") + winding_num;
                    });
                let mut weights = dvs
                    .iter()
                    .copied()
                    .map(|dv| {
                        let nv = v + dv;
                        neighbors
                            .iter()
                            .map(|x| (nv - x).abs())
                            .map(|x| pots.get(x as usize).copied().unwrap_or(f64::infinity()))
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();

                let min_e = weights
                    .iter()
                    .copied()
                    .fold(f64::infinity(), |acc, x| acc.min(x));
                weights.iter_mut().for_each(|x| *x = (-(*x - min_e)).exp());
                // Probs is now the set of boltzman weights for each of the dvs
                let total_weight = weights.iter().sum::<f64>();
                // Probs is now the acceptance probabilities for each of the dvs

                let mut rng = rand::thread_rng();
                let choice = rng.gen::<f64>() * total_weight;
                let dv = weights
                    .into_iter()
                    .zip(dvs.iter().copied())
                    .try_fold(choice, |mut acc, (w, dv)| {
                        acc -= w;
                        if acc <= 0.0 {
                            Err(dv)
                        } else {
                            Ok(acc)
                        }
                    })
                    .unwrap_err();
                *val += dv;
            });
    }
    fn old_update_b(&mut self) {
        let (wx, wy) = self.boundaries;
        let lx = self.a_lattice.shape()[0];
        let ly = 2 * self.a_lattice.shape()[1];
        let a_sub = self.a_lattice.view();
        let pots = self.potential.view();
        let dvs = &self.mc_trials;
        ndarray::Zip::indexed(&mut self.b_lattice)
            .into_par_iter()
            .for_each(|((ix, iy), val)| {
                let v = *val;
                let (x, y) = Self::b_to_global(ix, iy);
                let mut neighbors = [0; 4];
                neighbors
                    .iter_mut()
                    .zip(Self::neighbors(x, y, lx, ly).into_iter())
                    .for_each(|(v, (n, (wrap_x, wrap_y)))| {
                        let (ax, ay) = match n {
                            SubLatticeIndex::A(x, y) => (x, y),
                            SubLatticeIndex::B(_, _) => unreachable!(),
                        };
                        let winding_num = (wrap_x as i32 * wx) + (wrap_y as i32 * wy);
                        *v = *a_sub.get((ax, ay)).expect("neighbor not found") + winding_num;
                    });
                let mut weights = dvs
                    .iter()
                    .copied()
                    .map(|dv| {
                        let nv = v + dv;
                        neighbors
                            .iter()
                            .map(|x| (nv - x).abs())
                            .map(|x| pots.get(x as usize).copied().unwrap_or(f64::infinity()))
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                let min_e = weights
                    .iter()
                    .copied()
                    .fold(f64::infinity(), |acc, x| acc.min(x));
                weights.iter_mut().for_each(|x| *x = (-(*x - min_e)).exp());
                // Probs is now the set of boltzman weights for each of the dvs
                let total_weight = weights.iter().sum::<f64>();
                // Probs is now the acceptance probabilities for each of the dvs

                let mut rng = rand::thread_rng();
                let choice = rng.gen::<f64>() * total_weight;
                let dv = weights
                    .into_iter()
                    .zip(dvs.iter().copied())
                    .try_fold(choice, |mut acc, (w, dv)| {
                        acc -= w;
                        if acc <= 0.0 {
                            Err(dv)
                        } else {
                            Ok(acc)
                        }
                    })
                    .unwrap_err();
                *val += dv;
            });
    }
    pub fn get_total_energy(&self) -> f64 {
        let (wx, wy) = self.boundaries;
        let lx = self.a_lattice.shape()[0];
        let ly = 2 * self.a_lattice.shape()[1];
        ndarray::Zip::indexed(&self.a_lattice).fold(0.0, |acc, (ax, ay), v| {
            let b_neighbors = Self::a_neighbors(ax, ay, lx, ly);

            let res = b_neighbors
                .into_iter()
                .map(|((bx, by), (wrap_x, wrap_y))| {
                    let winding_num = (wrap_x as i32 * wx) + (wrap_y as i32 * wy);
                    let nv = self
                        .b_lattice
                        .get([bx, by])
                        .copied()
                        .expect("neighbor not found")
                        + winding_num;
                    self.potential
                        .get((nv - v).abs() as usize)
                        .copied()
                        .unwrap_or(f64::infinity())
                })
                .sum::<f64>();
            res + acc
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::SubLatticeIndex::{A, B};

    #[test]
    fn test_global_to_sub() {
        let sub = Lattice::global_to_sub(0, 0);
        assert_eq!(sub, A(0, 0));

        let sub = Lattice::global_to_sub(1, 0);
        assert_eq!(sub, B(1, 0));

        let sub = Lattice::global_to_sub(2, 0);
        assert_eq!(sub, A(2, 0));

        let sub = Lattice::global_to_sub(0, 1);
        assert_eq!(sub, B(0, 0));

        let sub = Lattice::global_to_sub(1, 1);
        assert_eq!(sub, A(1, 0));

        let sub = Lattice::global_to_sub(2, 1);
        assert_eq!(sub, B(2, 0));
    }

    #[test]
    fn test_sublat_neighbors() {
        for i in 0..4 {
            for j in 0..2 {
                let res = Lattice::a_neighbors(i, j, 4, 4);
                println!("{},{}\t{:?}", i, j, res)
            }
        }
    }

    #[test]
    fn test_neighbors() {
        let res = Lattice::neighbors(2, 2, 16, 16);
        assert_eq!(
            res,
            [
                (B(1, 1), (0, 0)),
                (B(3, 1), (0, 0)),
                (B(2, 0), (0, 0)),
                (B(2, 1), (0, 0))
            ]
        );
        let res = Lattice::neighbors(0, 0, 16, 16);
        assert_eq!(
            res,
            [
                (B(15, 0), (-1, 0)),
                (B(1, 0), (0, 0)),
                (B(0, 7), (0, -1)),
                (B(0, 0), (0, 0))
            ]
        );
        let res = Lattice::neighbors(1, 0, 16, 16);
        assert_eq!(
            res,
            [
                (A(0, 0), (0, 0)),
                (A(2, 0), (0, 0)),
                (A(1, 7), (0, -1)),
                (A(1, 0), (0, 0))
            ]
        );
        let res = Lattice::neighbors(15, 0, 16, 16);
        assert_eq!(
            res,
            [
                (A(14, 0), (0, 0)),
                (A(0, 0), (1, 0)),
                (A(15, 7), (0, -1)),
                (A(15, 0), (0, 0))
            ]
        );
    }

    #[test]
    fn it_works() {
        let L = 16;
        let mut result = Lattice::new(L, L, vec![0.0, 4.0, 1000.0], (0, 0));
        result.boundaries = (-1, 0);
        for _ in 0..100 {
            result.old_update();
        }

        for j in 0..L / 2 {
            for i in 0..L {
                print!("{}\t", result.a_lattice[[i, L / 2 - j - 1]]);
            }
            println!();
        }
        println!("====");

        for j in 0..L / 2 {
            for i in 0..L {
                print!("{}\t", result.b_lattice[[i, L / 2 - j - 1]]);
            }
            println!();
        }

        println!("====");

        for j in 0..L {
            for i in 0..L {
                print!("{}\t", result[&(i, L - j - 1)]);
            }
            println!();
        }

        println!("{:?}", result.get_energy(0, 0));

        println!("{:?}", result.get_total_energy());
    }
}
