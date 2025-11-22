use ratatui::text::Line;
use std::f32::consts::{FRAC_PI_4, TAU};
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

const BASE_HEAD_CENTER: Vec2 = Vec2 { x: -4.0, y: -6.0 };
const DEFAULT_HEAD_SPACING: f32 = 11.6;
const PAIR_HEAD_SPACING: f32 = 15.2;
const TRIPLET_HEAD_SPACING: f32 = 13.4;
const MULTI_VERTICAL_SWAY: f32 = 0.22;
const HEAD_STEP: f32 = 0.35;
const HEAD_RADIUS_X: f32 = 5.8;
const HEAD_RADIUS_Y: f32 = 3.9;
const STEM_STEP: f32 = 0.4;
const STEM_LEFT_OFFSET: f32 = 5.6;
const STEM_RIGHT_OFFSET: f32 = 7.0;
const STEM_BOTTOM_OFFSET: f32 = -0.5;
const STEM_TOP_SINGLE: f32 = 17.5;
const STEM_TOP_MULTI: f32 = 22.5;
const FLAG_CONTROL_OFFSETS: [Vec2; 4] = [
    Vec2 { x: 7.0, y: 17.5 },
    Vec2 { x: 15.5, y: 19.5 },
    Vec2 { x: 16.8, y: 13.0 },
    Vec2 { x: 10.8, y: 9.2 },
];
const FLAG_SEGMENTS: usize = 48;
const FLAG_THICKNESS: f32 = 2.7;
const FLAG_LAYERS: usize = 6;
const BEAM_STEP: f32 = 0.35;
const BEAM_DEPTH: f32 = 2.2;
const BEAM_BOTTOM_CLEARANCE: f32 = 4.2;
const BEAM_TOP_CLEARANCE: f32 = 1.1;
const DOT_STEP: f32 = 0.28;
const DOT_RADIUS: f32 = 1.45;
const DOT_OFFSET: Vec2 = Vec2 { x: 13.0, y: 0.8 };
const DOT_DEPTH: f32 = 0.8;

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
    Beam,
    Dot,
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

#[derive(Clone, Copy, Debug)]
pub enum NoteGlyph {
    SingleEighth,
    DottedEighth,
    JoinedPair,
    Triplet,
}

struct Bounds {
    extent_x: f32,
    extent_y: f32,
    extent_z: f32,
}

impl Bounds {
    fn max_extent(&self) -> f32 {
        self.extent_x.max(self.extent_y).max(self.extent_z)
    }
}

pub struct RotatingNoteGlyph {
    points: Vec<NotePoint>,
    light_dir: Vec3,
    projection_distance: f32,
    spin_speed: f32,
    tilt_speed: f32,
    ambient: f32,
    extent_x: f32,
    extent_y: f32,
    buffer: Vec<char>,
    depth: Vec<f32>,
}

impl RotatingNoteGlyph {
    pub fn new(variant: NoteGlyph) -> Self {
        let mut points = build_points(variant);
        let bounds = recenter(&mut points);
        let max_extent = bounds.max_extent().max(1.0);

        let projection_distance = max_extent * 2.6 + 14.0;
        let light_dir = Vec3::new(-0.35, 0.42, 1.0).normalized();

        Self {
            points,
            light_dir,
            projection_distance,
            spin_speed: TAU / 5.8,
            tilt_speed: TAU / 7.4,
            ambient: 0.24,
            extent_x: bounds.extent_x.max(1.0),
            extent_y: bounds.extent_y.max(1.0),
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
        let spin_angle = (t * self.spin_speed * 0.5).rem_euclid(TAU);
        let tilt_phase = t * self.tilt_speed;
        let tilt_angle = tilt_phase.sin() * FRAC_PI_4;
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

        let base_depth = self.projection_distance;
        let half_w = ((width as f32).max(2.0) - 2.0) * 0.5;
        let half_h = ((height as f32).max(2.0) - 2.0) * 0.5;
        let scale_x = half_w * base_depth / self.extent_x;
        let scale_y = half_h * base_depth / self.extent_y;
        let scale = scale_x.min(scale_y) * 1.08;

        let center_x = (width as f32 - 1.0) / 2.0;
        let center_y = (height as f32 - 1.0) / 2.0;

        for point in &self.points {
            let rotated_pos = rotate(point.position, sin_a, cos_a, sin_b, cos_b);
            let rotated_normal = rotate(point.normal, sin_a, cos_a, sin_b, cos_b).normalized();

            let depth_value = rotated_pos.z + base_depth;
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

fn build_points(variant: NoteGlyph) -> Vec<NotePoint> {
    let mut points = Vec::new();
    match variant {
        NoteGlyph::SingleEighth => build_single_eighth(&mut points),
        NoteGlyph::DottedEighth => build_dotted_eighth(&mut points),
        NoteGlyph::JoinedPair => build_joined_notes(&mut points, 2),
        NoteGlyph::Triplet => build_joined_notes(&mut points, 3),
    }
    points
}

fn build_single_eighth(points: &mut Vec<NotePoint>) {
    add_head(points, BASE_HEAD_CENTER);
    add_stem(points, BASE_HEAD_CENTER, STEM_TOP_SINGLE);
    add_flag(points, BASE_HEAD_CENTER);
}

fn build_dotted_eighth(points: &mut Vec<NotePoint>) {
    add_head(points, BASE_HEAD_CENTER);
    add_stem(points, BASE_HEAD_CENTER, STEM_TOP_SINGLE);
    add_flag(points, BASE_HEAD_CENTER);
    add_dot(points, BASE_HEAD_CENTER);
}

fn build_joined_notes(points: &mut Vec<NotePoint>, count: usize) {
    let spacing = match count {
        2 => PAIR_HEAD_SPACING,
        3 => TRIPLET_HEAD_SPACING,
        _ => DEFAULT_HEAD_SPACING,
    };
    let centers = head_centers(count, spacing);
    for center in &centers {
        add_head(points, *center);
        add_stem(points, *center, STEM_TOP_MULTI);
    }
    add_beam(points, &centers, STEM_TOP_MULTI);
}

fn head_centers(count: usize, spacing: f32) -> Vec<Vec2> {
    if count == 0 {
        return Vec::new();
    }

    let half = (count.saturating_sub(1) as f32) * 0.5;
    let mut centers = Vec::with_capacity(count);
    for i in 0..count {
        let offset = i as f32 - half;
        let x = BASE_HEAD_CENTER.x + offset * spacing;
        let y = BASE_HEAD_CENTER.y - offset * MULTI_VERTICAL_SWAY;
        centers.push(Vec2::new(x, y));
    }
    centers
}

fn add_head(points: &mut Vec<NotePoint>, center: Vec2) {
    let x_steps = ((HEAD_RADIUS_X * 2.0) / HEAD_STEP).ceil() as i32;
    let y_steps = ((HEAD_RADIUS_Y * 2.0) / HEAD_STEP).ceil() as i32;

    for iy in -y_steps..=y_steps {
        let y = center.y + iy as f32 * HEAD_STEP;
        for ix in -x_steps..=x_steps {
            let x = center.x + ix as f32 * HEAD_STEP;
            let dx = x - center.x;
            let dy = y - center.y;
            let normalized = (dx * dx) / (HEAD_RADIUS_X * HEAD_RADIUS_X)
                + (dy * dy) / (HEAD_RADIUS_Y * HEAD_RADIUS_Y);
            if normalized <= 1.0 {
                let edge = (1.0 - normalized).max(0.0);
                let depth = edge.sqrt();
                let z = depth.powf(0.85) * 3.2;
                let albedo = 0.68 + 0.25 * depth;
                let mut normal = Vec3::new(
                    dx / (HEAD_RADIUS_X * HEAD_RADIUS_X),
                    dy / (HEAD_RADIUS_Y * HEAD_RADIUS_Y),
                    depth / 1.2,
                );
                normal = normal.normalized();
                let position = Vec3::new(x, y, z);
                points.push(NotePoint::new(position, normal, albedo, NoteSegment::Head));
            }
        }
    }
}

fn add_stem(points: &mut Vec<NotePoint>, center: Vec2, top_offset: f32) {
    let left = center.x + STEM_LEFT_OFFSET;
    let right = center.x + STEM_RIGHT_OFFSET;
    let bottom = center.y + STEM_BOTTOM_OFFSET;
    let top = center.y + top_offset;

    let x_steps = ((right - left) / STEM_STEP).ceil() as i32;
    let y_steps = ((top - bottom) / STEM_STEP).ceil() as i32;

    for iy in 0..=y_steps {
        let y = bottom + iy as f32 * STEM_STEP;
        for ix in 0..=x_steps {
            let x = left + ix as f32 * STEM_STEP;
            let lateral = if x_steps == 0 { 0.0 } else { (ix as f32 / x_steps as f32 - 0.5).abs() };
            let thickness = (1.0 - lateral * 1.6).clamp(0.0, 1.0);
            let z = thickness * 2.2;
            let albedo = 0.56 + 0.18 * thickness;
            let mut normal = Vec3::new(0.9 - lateral * 0.4, 0.1, 1.6);
            normal = normal.normalized();
            let position = Vec3::new(x, y, z);
            points.push(NotePoint::new(position, normal, albedo, NoteSegment::Stem));
        }
    }
}

fn add_flag(points: &mut Vec<NotePoint>, center: Vec2) {
    let control = [
        Vec2::new(center.x + FLAG_CONTROL_OFFSETS[0].x, center.y + FLAG_CONTROL_OFFSETS[0].y),
        Vec2::new(center.x + FLAG_CONTROL_OFFSETS[1].x, center.y + FLAG_CONTROL_OFFSETS[1].y),
        Vec2::new(center.x + FLAG_CONTROL_OFFSETS[2].x, center.y + FLAG_CONTROL_OFFSETS[2].y),
        Vec2::new(center.x + FLAG_CONTROL_OFFSETS[3].x, center.y + FLAG_CONTROL_OFFSETS[3].y),
    ];
    add_flag_with_control(points, control);
}

fn add_flag_with_control(points: &mut Vec<NotePoint>, control: [Vec2; 4]) {
    for i in 0..=FLAG_SEGMENTS {
        let t = i as f32 / FLAG_SEGMENTS as f32;
        let center = cubic_point(control, t);
        let tangent = cubic_tangent(control, t).normalized();
        if tangent.length() == 0.0 {
            continue;
        }
        let plane_normal = Vec2::new(-tangent.y, tangent.x).normalized();

        for layer in 0..FLAG_LAYERS {
            let interp =
                if FLAG_LAYERS == 1 { 0.0 } else { layer as f32 / (FLAG_LAYERS - 1) as f32 };
            let offset = (interp - 0.5) * FLAG_THICKNESS;
            let position2d = center + plane_normal * offset;
            let distance = (offset / (FLAG_THICKNESS / 2.0)).abs().min(1.0);
            let albedo = 0.52 + 0.16 * (1.0 - distance);
            let z = (1.0 - distance.powi(2)) * 1.6;
            let mut normal = Vec3::new(plane_normal.x * 0.7, plane_normal.y * 0.7, 1.1 - distance);
            normal = normal.normalized();
            let position = Vec3::new(position2d.x, position2d.y, z);
            points.push(NotePoint::new(position, normal, albedo, NoteSegment::Flag));
        }
    }
}

fn add_beam(points: &mut Vec<NotePoint>, centers: &[Vec2], stem_top_offset: f32) {
    if centers.is_empty() {
        return;
    }

    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut avg_y = 0.0;
    for center in centers {
        min_x = min_x.min(center.x);
        max_x = max_x.max(center.x);
        avg_y += center.y;
    }
    avg_y /= centers.len() as f32;

    let left = min_x + STEM_LEFT_OFFSET - 2.2;
    let right = max_x + STEM_RIGHT_OFFSET + 2.6;
    let stem_top = avg_y + stem_top_offset;
    let bottom = stem_top - BEAM_BOTTOM_CLEARANCE;
    let top = stem_top - BEAM_TOP_CLEARANCE;
    let tilt = if centers.len() <= 1 {
        0.0
    } else {
        let first = centers.first().unwrap();
        let last = centers.last().unwrap();
        (last.y - first.y) * 0.35
    };

    add_beam_rect(points, left, right, bottom, top, tilt);
}

fn add_beam_rect(
    points: &mut Vec<NotePoint>,
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    tilt: f32,
) {
    if right <= left || top <= bottom {
        return;
    }

    let x_steps = ((right - left) / BEAM_STEP).ceil() as i32;
    let y_steps = ((top - bottom) / BEAM_STEP).ceil() as i32;

    for iy in 0..=y_steps {
        let base_y = bottom + iy as f32 * BEAM_STEP;
        let vertical = if y_steps == 0 { 0.0 } else { (iy as f32 / y_steps as f32 - 0.5).abs() };
        let vertical_falloff = (1.0 - vertical * 1.4).clamp(0.0, 1.0);
        let z = (1.0 - vertical * vertical) * BEAM_DEPTH;
        let albedo = 0.52 + 0.20 * vertical_falloff;

        for ix in 0..=x_steps {
            let ratio = if x_steps == 0 { 0.0 } else { ix as f32 / x_steps as f32 };
            let x = left + ix as f32 * BEAM_STEP;
            let y = base_y + tilt * (ratio - 0.5);
            let lateral = (ratio - 0.5).abs();
            let depth = z * (1.0 - lateral * 0.3) * vertical_falloff;
            let mut normal = Vec3::new(0.15 - lateral * 0.12, 0.35 - vertical * 0.15, 1.25);
            normal = normal.normalized();
            let position = Vec3::new(x, y, depth);
            points.push(NotePoint::new(position, normal, albedo, NoteSegment::Beam));
        }
    }
}

fn add_dot(points: &mut Vec<NotePoint>, base: Vec2) {
    let center = Vec2::new(base.x + DOT_OFFSET.x, base.y + DOT_OFFSET.y);
    let steps = ((DOT_RADIUS * 2.0) / DOT_STEP).ceil() as i32;

    for iy in -steps..=steps {
        let y = center.y + iy as f32 * DOT_STEP;
        for ix in -steps..=steps {
            let x = center.x + ix as f32 * DOT_STEP;
            let dx = x - center.x;
            let dy = y - center.y;
            let distance_sq = dx * dx + dy * dy;
            if distance_sq <= DOT_RADIUS * DOT_RADIUS {
                let normalized = (distance_sq / (DOT_RADIUS * DOT_RADIUS)).min(1.0);
                let depth = (1.0 - normalized).max(0.0);
                let z = depth.sqrt() * (DOT_RADIUS * 0.9) + DOT_DEPTH;
                let albedo = 0.62 + 0.22 * depth;
                let normal = Vec3::new(dx, dy, DOT_RADIUS).normalized();
                let position = Vec3::new(x, y, z);
                points.push(NotePoint::new(position, normal, albedo, NoteSegment::Dot));
            }
        }
    }
}

fn recenter(points: &mut [NotePoint]) -> Bounds {
    if points.is_empty() {
        return Bounds { extent_x: 1.0, extent_y: 1.0, extent_z: 1.0 };
    }

    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;

    for point in points.iter() {
        min_x = min_x.min(point.position.x);
        max_x = max_x.max(point.position.x);
        min_y = min_y.min(point.position.y);
        max_y = max_y.max(point.position.y);
        min_z = min_z.min(point.position.z);
        max_z = max_z.max(point.position.z);
    }

    let center_x = (max_x + min_x) * 0.5;
    let center_y = (max_y + min_y) * 0.5;
    let center_z = (max_z + min_z) * 0.5;

    for point in points.iter_mut() {
        point.position.x -= center_x;
        point.position.y -= center_y;
        point.position.z -= center_z;
    }

    Bounds {
        extent_x: (max_x - min_x) * 0.5,
        extent_y: (max_y - min_y) * 0.5,
        extent_z: (max_z - min_z) * 0.5,
    }
}

fn glyph_for(segment: NoteSegment, brightness: f32) -> char {
    match segment {
        NoteSegment::Head => palette_char(brightness, &['@', '#', 'O', 'o', '.', ' ']),
        NoteSegment::Stem => palette_char(brightness, &['|', '!', ':', '.', ' ']),
        NoteSegment::Flag => palette_char(brightness, &['~', '-', '`', '.', ' ']),
        NoteSegment::Beam => palette_char(brightness, &['=', '-', ':', '.', ' ']),
        NoteSegment::Dot => palette_char(brightness, &['*', '+', '.', ' ']),
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
    let a = (control[1] - control[0]) * (3.0 * u * u);
    let b = (control[2] - control[1]) * (6.0 * u * t);
    let c = (control[3] - control[2]) * (3.0 * t * t);
    a + b + c
}
