use ndarray::{Array1, Array2};
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
    check_range: usize,
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
    pub fn new<V: Into<Array1<f64>>>(x: usize, y: usize, potential: V) -> Self {
        assert_eq!(x % 2, 0);
        assert_eq!(y % 2, 0);
        Self {
            a_lattice: Array2::zeros((x, y / 2)),
            b_lattice: Array2::zeros((x, y / 2)),
            boundaries: (0, 0),
            potential: potential.into(),
            check_range: 1,
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

    pub fn update(&mut self) {
        self.update_a();
        self.update_b();
    }
    fn update_a(&mut self) {
        let (wx, wy) = self.boundaries;
        let lx = self.a_lattice.shape()[0];
        let ly = 2 * self.a_lattice.shape()[1];
        let b_sub = self.b_lattice.view();
        let pots = self.potential.view();
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
                let dvs = -(self.check_range as i32)..=self.check_range as i32;
                let mut weights = dvs
                    .clone()
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
                    .zip(dvs)
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
    fn update_b(&mut self) {
        let (wx, wy) = self.boundaries;
        let lx = self.a_lattice.shape()[0];
        let ly = 2 * self.a_lattice.shape()[1];
        let a_sub = self.a_lattice.view();
        let pots = self.potential.view();
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
                let dvs = -(self.check_range as i32)..=self.check_range as i32;
                let mut weights = dvs
                    .clone()
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
                    .zip(dvs)
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
        ndarray::Zip::indexed(&self.a_lattice)
            .into_par_iter()
            .map(|((ax, ay), v)| {
                let (x, y) = Self::a_to_global(ax, ay);
                let positive_neighbors = Self::neighbors(x, y, lx, ly);
                positive_neighbors
                    .into_iter()
                    .map(|(n, (wrap_x, wrap_y))| {
                        let winding_num = (wrap_x as i32 * wx) + (wrap_y as i32 * wy);
                        let nv = match n {
                            SubLatticeIndex::B(x, y) => self
                                .b_lattice
                                .get([x, y])
                                .copied()
                                .expect("neighbor not found"),
                            SubLatticeIndex::A(_, _) => unreachable!(),
                        } + winding_num;
                        self.potential
                            .get((nv - v).abs() as usize)
                            .copied()
                            .unwrap_or(f64::infinity())
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
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
        let mut result = Lattice::new(L, L, vec![0.0, 4.0, 1000.0]);
        result.boundaries = (-1, 0);
        for _ in 0..100 {
            result.update();
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
