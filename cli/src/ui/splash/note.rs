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

#[derive(Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn normalized(self) -> Self {
        let len = self.length();
        if len == 0.0 {
            Self::new(0.0, 0.0, 0.0)
        } else {
            Self::new(self.x / len, self.y / len, self.z / len)
        }
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
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
    position: Vec3,
    normal: Vec3,
    albedo: f32,
    segment: NoteSegment,
}

impl NotePoint {
    fn new(position: Vec3, normal: Vec3, albedo: f32, segment: NoteSegment) -> Self {
        Self { position, normal, albedo, segment }
    }
}

pub struct RotatingEighthNote {
    points: Vec<NotePoint>,
    light_dir: Vec3,
    projection_distance: f32,
    scale_factor: f32,
    spin_speed: f32,
    tilt_speed: f32,
    ambient: f32,
    buffer: Vec<char>,
    depth: Vec<f32>,
}

impl Default for RotatingEighthNote {
    fn default() -> Self {
        Self::new()
    }
}

impl RotatingEighthNote {
    pub fn new() -> Self {
        let mut points = build_points();
        recenter(&mut points);

        let radius = points.iter().map(|p| p.position.length()).fold(0.0_f32, f32::max).max(1.0);
        let light_dir = Vec3::new(-0.3, 0.45, 1.0).normalized();
        let projection_distance = radius * 4.0 + 10.0;
        let scale_factor = 0.9 / radius;
        let spin_speed = TAU / 6.5;
        let tilt_speed = TAU / 8.5;
        let ambient = 0.22;

        Self {
            points,
            light_dir,
            projection_distance,
            scale_factor,
            spin_speed,
            tilt_speed,
            ambient,
            buffer: Vec::new(),
            depth: Vec::new(),
        }
    }

    pub fn render(&mut self, width: u16, height: u16, elapsed: StdDuration) -> Vec<Line<'static>> {
        let width = width as usize;
        let height = height as usize;
        if width == 0 || height == 0 {
            return Vec::new();
        }

        let t = elapsed.as_secs_f32();
        let spin_angle = (t * self.spin_speed).rem_euclid(TAU);
        let tilt_angle = (t * self.tilt_speed).rem_euclid(TAU);
        let (sin_a, cos_a) = spin_angle.sin_cos();
        let (sin_b, cos_b) = tilt_angle.sin_cos();

        let buffer_len = width * height;
        if self.buffer.len() != buffer_len {
            self.buffer = vec![' '; buffer_len];
        } else {
            self.buffer.fill(' ');
        }
        if self.depth.len() != buffer_len {
            self.depth = vec![f32::INFINITY; buffer_len];
        } else {
            self.depth.fill(f32::INFINITY);
        }

        let center_x = (width as f32 - 1.0) / 2.0;
        let center_y = (height as f32 - 1.0) / 2.0;
        let scale = self.scale_factor * (width.min(height) as f32);

        for point in &self.points {
            let rotated_pos = rotate(point.position, sin_a, cos_a, sin_b, cos_b);
            let rotated_normal = rotate(point.normal, sin_a, cos_a, sin_b, cos_b).normalized();

            let depth_value = rotated_pos.z + self.projection_distance;
            if depth_value <= 1.0 {
                continue;
            }

            let inv_depth = 1.0 / depth_value;
            let projected_x = rotated_pos.x * scale * inv_depth + center_x;
            let projected_y = -rotated_pos.y * scale * inv_depth + center_y;

            let x = projected_x.round() as isize;
            let y = projected_y.round() as isize;
            if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
                continue;
            }

            let diffuse = rotated_normal.dot(self.light_dir).max(0.0);
            let brightness = (self.ambient + diffuse * point.albedo).clamp(0.0, 1.0);
            if brightness <= 0.05 {
                continue;
            }

            let idx = y as usize * width + x as usize;
            if depth_value < self.depth[idx] {
                self.depth[idx] = depth_value;
                self.buffer[idx] = glyph_for(point.segment, brightness);
            }
        }

        let mut lines = Vec::with_capacity(height);
        for row in self.buffer.chunks(width) {
            let mut line = String::with_capacity(width);
            for ch in row {
                line.push(*ch);
            }
            lines.push(Line::from(line));
        }
        lines
    }
}

fn rotate(point: Vec3, sin_a: f32, cos_a: f32, sin_b: f32, cos_b: f32) -> Vec3 {
    let x = point.x * cos_a - point.y * sin_a;
    let y = point.x * sin_a + point.y * cos_a;
    let z = point.z;

    Vec3::new(x * cos_b + z * sin_b, y, -x * sin_b + z * cos_b)
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
            let dx = x - CENTER.x;
            let dy = y - CENTER.y;
            let normalized = (dx * dx) / (RADIUS_X * RADIUS_X) + (dy * dy) / (RADIUS_Y * RADIUS_Y);
            if normalized <= 1.0 {
                let edge_falloff = (1.0 - normalized).max(0.0).sqrt();
                let albedo = 0.65 + 0.3 * edge_falloff;
                let mut normal =
                    Vec3::new(dx / (RADIUS_X * RADIUS_X), dy / (RADIUS_Y * RADIUS_Y), 0.25);
                normal = normal.normalized();
                let position = Vec3::new(x, y, 0.0);
                points.push(NotePoint::new(position, normal, albedo, NoteSegment::Head));
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
            let albedo = 0.55 + 0.2 * (1.0 - lateral * 1.6).clamp(0.0, 1.0);
            let mut normal = Vec3::new(1.0, 0.3 - lateral * 0.6, 0.3);
            normal = normal.normalized();
            let position = Vec3::new(x, y, 0.0);
            points.push(NotePoint::new(position, normal, albedo, NoteSegment::Stem));
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
        let plane_normal = Vec2::new(-tangent.y, tangent.x).normalized();

        for layer in 0..LAYERS {
            let interp = if LAYERS == 1 { 0.0 } else { layer as f32 / (LAYERS - 1) as f32 };
            let offset = (interp - 0.5) * THICKNESS;
            let position2d = center + plane_normal * offset;
            let distance = (offset / (THICKNESS / 2.0)).abs().min(1.0);
            let albedo = 0.5 + 0.18 * (1.0 - distance);
            let mut normal = Vec3::new(plane_normal.x, plane_normal.y, 0.2 - distance * 0.3);
            normal = normal.normalized();
            let position = Vec3::new(position2d.x, position2d.y, 0.0);
            points.push(NotePoint::new(position, normal, albedo, NoteSegment::Flag));
        }
    }
}

fn recenter(points: &mut [NotePoint]) {
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
}

fn glyph_for(segment: NoteSegment, brightness: f32) -> char {
    match segment {
        NoteSegment::Head => palette_char(brightness, &['@', '#', 'O', 'o', '.', ' ']),
        NoteSegment::Stem => palette_char(brightness, &['|', '!', ':', '.']),
        NoteSegment::Flag => palette_char(brightness, &['~', '-', '`', '.', ' ']),
    }
}

fn palette_char(brightness: f32, palette: &[char]) -> char {
    let normalized = brightness.clamp(0.0, 0.999);
    let levels = palette.len().saturating_sub(1) as f32;
    let index = (levels * (1.0 - normalized)).round() as usize;
    palette[index.min(palette.len() - 1)]
}

fn cubic_point(control: [Vec2; 4], t: f32) -> Vec2 {
    let u = 1.0 - t;
    let u2 = u * u;
    let t2 = t * t;
    Vec2::new(
        u2 * u * control[0].x
            + 3.0 * u2 * t * control[1].x
            + 3.0 * u * t2 * control[2].x
            + t2 * t * control[3].x,
        u2 * u * control[0].y
            + 3.0 * u2 * t * control[1].y
            + 3.0 * u * t2 * control[2].y
            + t2 * t * control[3].y,
    )
}

fn cubic_tangent(control: [Vec2; 4], t: f32) -> Vec2 {
    let u = 1.0 - t;
    let a = (control[1] - control[0]) * 3.0 * u * u;
    let b = (control[2] - control[1]) * 6.0 * u * t;
    let c = (control[3] - control[2]) * 3.0 * t * t;
    Vec2::new(a.x + b.x + c.x, a.y + b.y + c.y)
}
