use num_traits::Float;

use crate::scaling::ScalerError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect<T> {
    pub x_min: T,
    pub x_max: T,
    pub y_min: T,
    pub y_max: T,
}

impl<T> Rect<T>
where
    T: Float,
{
    pub fn new(x_min: T, x_max: T, y_min: T, y_max: T) -> Result<Self, ScalerError<T>> {
        let rect = Self {
            x_min,
            x_max,
            y_min,
            y_max,
        };

        if !x_min.is_finite()
            || !x_max.is_finite()
            || !y_min.is_finite()
            || !y_max.is_finite()
            || x_min >= x_max
            || y_min >= y_max
        {
            return Err(ScalerError::InvalidRect(rect));
        }

        Ok(rect)
    }

    pub fn width(&self) -> T {
        self.x_max - self.x_min
    }

    pub fn height(&self) -> T {
        self.y_max - self.y_min
    }

    pub fn contains(&self, p: Point<T>) -> bool {
        self.x_min <= p.x && p.x <= self.x_max && self.y_min <= p.y && p.y <= self.y_max
    }

    pub fn centre(&self) -> Point<T> {
        let two = T::one() + T::one();

        Point {
            x: (self.x_min + self.x_max) / two,
            y: (self.y_min + self.y_max) / two,
        }
    }

    pub fn area(&self) -> T {
        self.width() * self.height()
    }

    pub fn quadrants(&self) -> Result<[Rect<T>; 4], ScalerError<T>> {
        let c = self.centre();

        Ok([
            Rect::new(self.x_min, c.x, self.y_min, c.y)?,
            Rect::new(c.x, self.x_max, self.y_min, c.y)?,
            Rect::new(self.x_min, c.x, c.y, self.y_max)?,
            Rect::new(c.x, self.x_max, c.y, self.y_max)?,
        ])
    }

    pub fn aspect_ratio(&self) -> T {
        self.width() / self.height()
    }

    pub fn longest_side(&self) -> T
    where
        T: Float,
    {
        self.width().max(self.height())
    }

    pub fn shortest_side(&self) -> T
    where
        T: Float,
    {
        self.width().min(self.height())
    }

    pub fn diameter(&self) -> T
    where
        T: Float,
    {
        let w = self.width();
        let h = self.height();

        (w * w + h * h).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const TOL: f64 = 1e-12;

    #[test]
    fn rect_new_accepts_valid_rectangle() {
        let rect = Rect::new(0.0, 2.0, -1.0, 1.0).unwrap();

        assert_relative_eq!(rect.width(), 2.0, epsilon = TOL);
        assert_relative_eq!(rect.height(), 2.0, epsilon = TOL);
        assert_relative_eq!(rect.area(), 4.0, epsilon = TOL);
    }

    #[test]
    fn rect_new_rejects_zero_width() {
        assert!(matches!(
            Rect::new(1.0, 1.0, 0.0, 1.0),
            Err(ScalerError::InvalidRect(_))
        ));
    }

    #[test]
    fn rect_new_rejects_zero_height() {
        assert!(matches!(
            Rect::new(0.0, 1.0, 1.0, 1.0),
            Err(ScalerError::InvalidRect(_))
        ));
    }

    #[test]
    fn rect_new_rejects_reversed_x_bounds() {
        assert!(matches!(
            Rect::new(2.0, 1.0, 0.0, 1.0),
            Err(ScalerError::InvalidRect(_))
        ));
    }

    #[test]
    fn rect_new_rejects_reversed_y_bounds() {
        assert!(matches!(
            Rect::new(0.0, 1.0, 2.0, 1.0),
            Err(ScalerError::InvalidRect(_))
        ));
    }

    #[test]
    fn rect_new_rejects_nan() {
        assert!(matches!(
            Rect::new(f64::NAN, 1.0, 0.0, 1.0),
            Err(ScalerError::InvalidRect(_))
        ));
    }

    #[test]
    fn rect_new_rejects_infinity() {
        assert!(matches!(
            Rect::new(0.0, f64::INFINITY, 0.0, 1.0),
            Err(ScalerError::InvalidRect(_))
        ));
    }

    #[test]
    fn rect_contains_includes_boundary() {
        let rect = Rect::new(0.0, 1.0, 0.0, 1.0).unwrap();

        assert!(rect.contains(Point { x: 0.0, y: 0.0 }));
        assert!(rect.contains(Point { x: 1.0, y: 1.0 }));
        assert!(rect.contains(Point { x: 0.5, y: 0.5 }));
    }

    #[test]
    fn rect_contains_rejects_outside_points() {
        let rect = Rect::new(0.0, 1.0, 0.0, 1.0).unwrap();

        assert!(!rect.contains(Point { x: -0.1, y: 0.5 }));
        assert!(!rect.contains(Point { x: 0.5, y: 1.1 }));
    }

    #[test]
    fn rect_centre_is_midpoint() {
        let rect = Rect::new(-2.0, 4.0, 10.0, 20.0).unwrap();
        let c = rect.centre();

        assert_relative_eq!(c.x, 1.0, epsilon = TOL);
        assert_relative_eq!(c.y, 15.0, epsilon = TOL);
    }

    #[test]
    fn rect_quadrants_cover_parent_area() {
        let rect = Rect::new(0.0, 4.0, 0.0, 2.0).unwrap();
        let qs = rect.quadrants().unwrap();

        let area: f64 = qs.iter().map(|q| q.area()).sum();

        assert_relative_eq!(area, rect.area(), epsilon = TOL);

        for q in qs {
            assert!(rect.contains(Point {
                x: q.x_min,
                y: q.y_min
            }));
            assert!(rect.contains(Point {
                x: q.x_max,
                y: q.y_max
            }));
        }
    }

    #[test]
    fn rect_quadrants_have_expected_bounds() {
        let rect = Rect::new(0.0, 2.0, 0.0, 2.0).unwrap();
        let qs = rect.quadrants().unwrap();

        assert_eq!(qs[0], Rect::new(0.0, 1.0, 0.0, 1.0).unwrap());
        assert_eq!(qs[1], Rect::new(1.0, 2.0, 0.0, 1.0).unwrap());
        assert_eq!(qs[2], Rect::new(0.0, 1.0, 1.0, 2.0).unwrap());
        assert_eq!(qs[3], Rect::new(1.0, 2.0, 1.0, 2.0).unwrap());
    }

    #[test]
    fn rect_longest_side_returns_larger_dimension() {
        let rect = Rect::new(0.0, 3.0, 0.0, 4.0).unwrap();

        assert_relative_eq!(rect.longest_side(), 4.0, epsilon = TOL);
    }

    #[test]
    fn rect_shortest_side_returns_smaller_dimension() {
        let rect = Rect::new(0.0, 3.0, 0.0, 4.0).unwrap();

        assert_relative_eq!(rect.shortest_side(), 3.0, epsilon = TOL);
    }

    #[test]
    fn rect_diameter_returns_diagonal_length() {
        let rect = Rect::new(0.0, 3.0, 0.0, 4.0).unwrap();

        assert_relative_eq!(rect.diameter(), 5.0, epsilon = TOL);
    }

    #[test]
    fn rect_square_has_equal_longest_and_shortest_side() {
        let rect = Rect::new(-1.0, 1.0, -2.0, 0.0).unwrap();

        assert_relative_eq!(rect.longest_side(), 2.0, epsilon = TOL);
        assert_relative_eq!(rect.shortest_side(), 2.0, epsilon = TOL);
        assert_relative_eq!(rect.diameter(), (8.0_f64).sqrt(), epsilon = TOL);
    }
}
