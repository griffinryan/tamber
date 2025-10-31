use ratatui::text::Line;
use std::f32::consts::TAU;
use std::time::Duration as StdDuration;

#[derive(Clone, Copy)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn normalized(self) -> Self {
        let len = self.length();
        if len == 0.0 {
            Self::new(0.0, 0.0)
        } else {
            Self::new(self.x / len, self.y / len)
        }
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl std::ops::Mul<Vec2> for f32 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2::new(rhs.x * self, rhs.y * self)
    }
}

#[derive(Clone, Copy)]
enum NoteSegment {
    Head,
    Stem,
    Flag,
}

#[derive(Clone, Copy)]
struct NotePoint {
    position: Vec2,
    intensity: f32,
    segment: NoteSegment,
}

impl NotePoint {
    fn new(position: Vec2, intensity: f32, segment: NoteSegment) -> Self {
        Self { position, intensity, segment }
    }
}

pub struct RotatingEighthNote {
    points: Vec<NotePoint>,
    extent_x: f32,
    extent_y: f32,
    radius: f32,
    light_dir: Vec2,
}

impl Default for RotatingEighthNote {
    fn default() -> Self {
        Self::new()
    }
}

impl RotatingEighthNote {
    pub fn new() -> Self {
        let mut points = build_points();
        let bounds = recenter_points(&mut points);
        let radius = points.iter().map(|p| p.position.length()).fold(0.0_f32, f32::max);
        let light_dir = Vec2::new(0.35, -1.0).normalized();
        Self {
            points,
            extent_x: bounds.extent_x,
            extent_y: bounds.extent_y,
            radius: radius.max(1.0),
            light_dir,
        }
    }

    pub fn render(&self, width: u16, height: u16, elapsed: StdDuration) -> Vec<Line<'static>> {
        let width = width as usize;
        let height = height as usize;
        if width == 0 || height == 0 {
            return Vec::new();
        }

        let mut buffer = vec![vec![' '; width]; height];
        let mut occupancy = vec![vec![0.0_f32; width]; height];

        let angle = (elapsed.as_secs_f32() * (TAU / 14.0)).rem_euclid(TAU);
        let sin_a = angle.sin();
        let cos_a = angle.cos();

        let width_f = width as f32;
        let height_f = height as f32;
        if width_f <= 1.0 || height_f <= 1.0 {
            return buffer
                .into_iter()
                .map(|row| Line::from(row.into_iter().collect::<String>()))
                .collect();
        }

        let margin = 1.08;
        let aspect = 0.62;

        let scale_x = if self.extent_x > 0.0 {
            (width_f - 1.0) / (self.extent_x * 2.0 * margin)
        } else {
            1.0
        };
        let scale_y = if self.extent_y > 0.0 {
            (height_f - 1.0) / (self.extent_y * 2.0 * margin * aspect)
        } else {
            1.0
        };
        let scale = scale_x.min(scale_y);
        let center_x = (width_f - 1.0) / 2.0;
        let center_y = (height_f - 1.0) / 2.0;

        for point in &self.points {
            let rotated_x = point.position.x * cos_a - point.position.y * sin_a;
            let rotated_y = point.position.x * sin_a + point.position.y * cos_a;

            let screen_x = rotated_x * scale + center_x;
            let screen_y = rotated_y * scale * aspect + center_y;

            let x = screen_x.round() as isize;
            let y = screen_y.round() as isize;
            if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
                continue;
            }

            let light = (Vec2::new(rotated_x, rotated_y).dot(self.light_dir)) / self.radius;
            let shading = (0.6 + 0.4 * light).clamp(0.2, 1.0);
            let brightness = (point.intensity * shading).clamp(0.0, 1.0);
            if brightness < 0.05 {
                continue;
            }

            let current = occupancy[y as usize][x as usize];
            if brightness > current {
                let glyph = glyph_for(point.segment, brightness);
                buffer[y as usize][x as usize] = glyph;
                occupancy[y as usize][x as usize] = brightness;
            }
        }

        buffer.into_iter().map(|row| Line::from(row.into_iter().collect::<String>())).collect()
    }
}

struct Bounds {
    extent_x: f32,
    extent_y: f32,
}

fn build_points() -> Vec<NotePoint> {
    let mut points = Vec::new();
    build_head(&mut points);
    build_stem(&mut points);
    build_flag(&mut points);
    points
}

fn build_head(points: &mut Vec<NotePoint>) {
    const STEP: f32 = 0.5;
    const CENTER: Vec2 = Vec2 { x: -4.0, y: -6.0 };
    const RADIUS_X: f32 = 5.8;
    const RADIUS_Y: f32 = 3.9;

    let x_steps = ((RADIUS_X * 2.0) / STEP).ceil() as i32;
    let y_steps = ((RADIUS_Y * 2.0) / STEP).ceil() as i32;

    for iy in -y_steps..=y_steps {
        let y = CENTER.y + iy as f32 * STEP;
        for ix in -x_steps..=x_steps {
            let x = CENTER.x + ix as f32 * STEP;
            let dx = (x - CENTER.x) / RADIUS_X;
            let dy = (y - CENTER.y) / RADIUS_Y;
            let distance = dx * dx + dy * dy;
            if distance <= 1.0 {
                let edge_falloff = (1.0 - distance).max(0.0).sqrt();
                let intensity = 0.6 + 0.38 * edge_falloff;
                points.push(NotePoint::new(Vec2::new(x, y), intensity, NoteSegment::Head));
            }
        }
    }
}

fn build_stem(points: &mut Vec<NotePoint>) {
    const STEP: f32 = 0.5;
    const LEFT: f32 = 1.6;
    const RIGHT: f32 = 3.0;
    const BOTTOM: f32 = -6.5;
    const TOP: f32 = 11.5;

    let x_steps = ((RIGHT - LEFT) / STEP).ceil() as i32;
    let y_steps = ((TOP - BOTTOM) / STEP).ceil() as i32;

    for iy in 0..=y_steps {
        let y = BOTTOM + iy as f32 * STEP;
        for ix in 0..=x_steps {
            let x = LEFT + ix as f32 * STEP;
            let lateral = if x_steps == 0 { 0.0 } else { (ix as f32 / x_steps as f32 - 0.5).abs() };
            let intensity = 0.55 + 0.25 * (1.0 - lateral * 1.8).clamp(0.0, 1.0);
            points.push(NotePoint::new(Vec2::new(x, y), intensity, NoteSegment::Stem));
        }
    }
}

fn build_flag(points: &mut Vec<NotePoint>) {
    const SEGMENTS: usize = 42;
    const THICKNESS: f32 = 2.6;
    const LAYERS: usize = 5;

    let control =
        [Vec2::new(3.0, 11.5), Vec2::new(11.5, 13.5), Vec2::new(12.8, 7.0), Vec2::new(6.8, 3.2)];

    for i in 0..=SEGMENTS {
        let t = i as f32 / SEGMENTS as f32;
        let center = cubic_point(control, t);
        let tangent = cubic_tangent(control, t).normalized();
        if tangent.length() == 0.0 {
            continue;
        }
        let normal = Vec2::new(-tangent.y, tangent.x);

        for layer in 0..LAYERS {
            let interp = if LAYERS == 1 { 0.0 } else { layer as f32 / (LAYERS - 1) as f32 };
            let offset = (interp - 0.5) * THICKNESS;
            let position = center + normal * offset;
            let distance = (offset / (THICKNESS / 2.0)).abs().min(1.0);
            let intensity = 0.45 + 0.18 * (1.0 - distance);
            points.push(NotePoint::new(position, intensity, NoteSegment::Flag));
        }
    }
}

fn recenter_points(points: &mut [NotePoint]) -> Bounds {
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    for point in points.iter() {
        min_x = min_x.min(point.position.x);
        max_x = max_x.max(point.position.x);
        min_y = min_y.min(point.position.y);
        max_y = max_y.max(point.position.y);
    }

    let center_x = (max_x + min_x) / 2.0;
    let center_y = (max_y + min_y) / 2.0;

    for point in points.iter_mut() {
        point.position.x -= center_x;
        point.position.y -= center_y;
    }

    Bounds { extent_x: (max_x - min_x) / 2.0, extent_y: (max_y - min_y) / 2.0 }
}

fn glyph_for(segment: NoteSegment, brightness: f32) -> char {
    match segment {
        NoteSegment::Head => palette_char(brightness, &['@', '#', 'O', 'o', '.']),
        NoteSegment::Stem => palette_char(brightness, &['|', '!', ':']),
        NoteSegment::Flag => palette_char(brightness, &['~', '-', '`', '.']),
    }
}

fn palette_char(brightness: f32, palette: &[char]) -> char {
    let normalized = brightness.clamp(0.0, 0.999);
    let levels = palette.len().saturating_sub(1) as f32;
    let index = (levels - levels * normalized).round() as usize;
    palette[index.min(palette.len() - 1)]
}

fn cubic_point(control: [Vec2; 4], t: f32) -> Vec2 {
    let u = 1.0 - t;
    let u2 = u * u;
    let t2 = t * t;
    u2 * u * control[0]
        + 3.0 * u2 * t * control[1]
        + 3.0 * u * t2 * control[2]
        + t2 * t * control[3]
}

fn cubic_tangent(control: [Vec2; 4], t: f32) -> Vec2 {
    let u = 1.0 - t;
    let a = 3.0 * (control[1] - control[0]) * u * u;
    let b = 6.0 * (control[2] - control[1]) * u * t;
    let c = 3.0 * (control[3] - control[2]) * t * t;
    a + b + c
}
