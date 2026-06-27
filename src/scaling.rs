//! Coordinate scaling for quadtree refinement.
//!
//! The quadtree is represented internally on a scaled rectangular domain,
//! usually the unit square `[0, 1] × [0, 1]`. User oracles are evaluated on the
//! original raw coordinate domain.
//!
//! [`Scaler2D`] provides the bijection between these coordinate systems.
//! Scaling is purely geometric: it is not part of the refinement criterion or
//! oracle model.

use crate::geometry::{Point, Rect};

use num_traits::Float;

#[derive(thiserror::Error, Debug)]
pub enum ScalerError<T> {
    #[error("invalid rectangle: {0:?}")]
    InvalidRect(Rect<T>),

    #[error("point {point:?} outside rectangle {rect:?}")]
    PointOutsideRect { point: Point<T>, rect: Rect<T> },
}

#[derive(Debug, Clone)]
pub struct Scaler2D<T> {
    raw: Rect<T>,
    scaled: Rect<T>,
}

impl<T> Scaler2D<T>
where
    T: Float,
{
    pub fn unit_square(raw: Rect<T>) -> Result<Self, ScalerError<T>> {
        let scaled = Rect::new(T::zero(), T::one(), T::zero(), T::one())?;
        Ok(Self { raw, scaled })
    }

    pub fn new(raw: Rect<T>, scaled: Rect<T>) -> Self {
        Self { raw, scaled }
    }

    pub fn raw_domain(&self) -> Rect<T>
    where
        T: Copy,
    {
        self.raw
    }

    pub fn scaled_domain(&self) -> Rect<T>
    where
        T: Copy,
    {
        self.scaled
    }

    pub fn to_raw(&self, p: Point<T>) -> Result<Point<T>, ScalerError<T>> {
        if !self.scaled.contains(p) {
            return Err(ScalerError::PointOutsideRect {
                point: p,
                rect: self.scaled,
            });
        }

        let ux = (p.x - self.scaled.x_min) / self.scaled.width();
        let uy = (p.y - self.scaled.y_min) / self.scaled.height();

        Ok(Point {
            x: self.raw.x_min + ux * self.raw.width(),
            y: self.raw.y_min + uy * self.raw.height(),
        })
    }

    pub fn to_scaled(&self, p: Point<T>) -> Result<Point<T>, ScalerError<T>> {
        if !self.raw.contains(p) {
            return Err(ScalerError::PointOutsideRect {
                point: p,
                rect: self.raw,
            });
        }

        let ux = (p.x - self.raw.x_min) / self.raw.width();
        let uy = (p.y - self.raw.y_min) / self.raw.height();

        Ok(Point {
            x: self.scaled.x_min + ux * self.scaled.width(),
            y: self.scaled.y_min + uy * self.scaled.height(),
        })
    }

    pub fn to_raw_rect(&self, rect: Rect<T>) -> Result<Rect<T>, ScalerError<T>> {
        let lower = self.to_raw(Point {
            x: rect.x_min,
            y: rect.y_min,
        })?;

        let upper = self.to_raw(Point {
            x: rect.x_max,
            y: rect.y_max,
        })?;

        Rect::new(lower.x, upper.x, lower.y, upper.y)
    }

    pub fn to_scaled_rect(&self, rect: Rect<T>) -> Result<Rect<T>, ScalerError<T>> {
        let lower = self.to_scaled(Point {
            x: rect.x_min,
            y: rect.y_min,
        })?;

        let upper = self.to_scaled(Point {
            x: rect.x_max,
            y: rect.y_max,
        })?;

        Rect::new(lower.x, upper.x, lower.y, upper.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const TOL: f64 = 1e-12;

    #[test]
    fn scaler_maps_unit_square_corners_to_raw_corners() {
        let raw = Rect::new(10.0, 20.0, -5.0, 5.0).unwrap();
        let scaler = Scaler2D::unit_square(raw).unwrap();

        assert_eq!(
            scaler.to_raw(Point { x: 0.0, y: 0.0 }).unwrap(),
            Point { x: 10.0, y: -5.0 }
        );

        assert_eq!(
            scaler.to_raw(Point { x: 1.0, y: 1.0 }).unwrap(),
            Point { x: 20.0, y: 5.0 }
        );
    }

    #[test]
    fn scaler_maps_unit_square_centre_to_raw_centre() {
        let raw = Rect::new(10.0, 20.0, -5.0, 5.0).unwrap();
        let scaler = Scaler2D::unit_square(raw).unwrap();

        let p = scaler.to_raw(Point { x: 0.5, y: 0.5 }).unwrap();

        assert_relative_eq!(p.x, 15.0, epsilon = TOL);
        assert_relative_eq!(p.y, 0.0, epsilon = TOL);
    }

    #[test]
    fn scaler_to_scaled_maps_raw_centre_to_unit_centre() {
        let raw = Rect::new(10.0, 20.0, -5.0, 5.0).unwrap();
        let scaler = Scaler2D::unit_square(raw).unwrap();

        let p = scaler.to_scaled(Point { x: 15.0, y: 0.0 }).unwrap();

        assert_relative_eq!(p.x, 0.5, epsilon = TOL);
        assert_relative_eq!(p.y, 0.5, epsilon = TOL);
    }

    #[test]
    fn scaler_supports_non_unit_scaled_domain() {
        let raw = Rect::new(0.0, 10.0, 0.0, 20.0).unwrap();
        let scaled = Rect::new(-1.0, 1.0, -2.0, 2.0).unwrap();
        let scaler = Scaler2D::new(raw, scaled);

        let raw_mid = scaler.to_raw(Point { x: 0.0, y: 0.0 }).unwrap();

        assert_relative_eq!(raw_mid.x, 5.0, epsilon = TOL);
        assert_relative_eq!(raw_mid.y, 10.0, epsilon = TOL);

        let scaled_mid = scaler.to_scaled(Point { x: 5.0, y: 10.0 }).unwrap();

        assert_relative_eq!(scaled_mid.x, 0.0, epsilon = TOL);
        assert_relative_eq!(scaled_mid.y, 0.0, epsilon = TOL);
    }

    #[test]
    fn scaler_rejects_scaled_point_outside_domain() {
        let raw = Rect::new(0.0, 10.0, 0.0, 10.0).unwrap();
        let scaler = Scaler2D::unit_square(raw).unwrap();

        assert!(matches!(
            scaler.to_raw(Point { x: -0.1, y: 0.5 }),
            Err(ScalerError::PointOutsideRect { .. })
        ));
    }

    #[test]
    fn scaler_rejects_raw_point_outside_domain() {
        let raw = Rect::new(0.0, 10.0, 0.0, 10.0).unwrap();
        let scaler = Scaler2D::unit_square(raw).unwrap();

        assert!(matches!(
            scaler.to_scaled(Point { x: 11.0, y: 5.0 }),
            Err(ScalerError::PointOutsideRect { .. })
        ));
    }

    #[test]
    fn scaler_roundtrip_scaled_to_raw_to_scaled() {
        let raw = Rect::new(-10.0, 30.0, 100.0, 200.0).unwrap();
        let scaler = Scaler2D::unit_square(raw).unwrap();

        let points = [
            Point { x: 0.0, y: 0.0 },
            Point { x: 0.25, y: 0.75 },
            Point { x: 0.5, y: 0.5 },
            Point { x: 1.0, y: 1.0 },
        ];

        for p in points {
            let raw = scaler.to_raw(p).unwrap();
            let back = scaler.to_scaled(raw).unwrap();

            assert_relative_eq!(back.x, p.x);
            assert_relative_eq!(back.y, p.y);
        }
    }

    #[test]
    fn scaler_roundtrip_raw_to_scaled_to_raw() {
        let raw = Rect::new(-10.0, 30.0, 100.0, 200.0).unwrap();
        let scaler = Scaler2D::unit_square(raw).unwrap();

        let points = [
            Point { x: -10.0, y: 100.0 },
            Point { x: 0.0, y: 125.0 },
            Point { x: 10.0, y: 150.0 },
            Point { x: 30.0, y: 200.0 },
        ];

        for p in points {
            let scaled = scaler.to_scaled(p).unwrap();
            let back = scaler.to_raw(scaled).unwrap();

            assert_relative_eq!(back.x, p.x);
            assert_relative_eq!(back.y, p.y);
        }
    }
}
