import SwiftUI

struct AnimatedNoteView: View {
    enum Intensity {
        case idle
        case active

        var gradient: [Color] {
            switch self {
            case .idle:
                return [
                    Color(red: 0.28, green: 0.62, blue: 0.96),
                    Color(red: 0.15, green: 0.25, blue: 0.56)
                ]
            case .active:
                return [
                    Color(red: 0.45, green: 0.86, blue: 0.98),
                    Color(red: 0.88, green: 0.62, blue: 0.98)
                ]
            }
        }
    }

    var intensity: Intensity = .idle
    @State private var rotation: Double = 0
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            DesignTokens.ColorToken.accent.opacity(0.5),
                            Color.white.opacity(0.05)
                        ],
                        center: .center,
                        startRadius: 10,
                        endRadius: 140
                    )
                )
                .blur(radius: 30)

            Circle()
                .fill(.ultraThinMaterial)
                .overlay(
                    Circle()
                        .strokeBorder(
                            AngularGradient(
                                colors: intensity.gradient,
                                center: .center
                            ),
                            lineWidth: 3
                        )
                        .blur(radius: 1)
                        .opacity(0.9)
                )
                .frame(width: 200, height: 200)
                .shadow(color: DesignTokens.ColorToken.accent.opacity(0.5), radius: 30, x: 0, y: 18)
                .overlay(noteGlyph)
                .overlay(particles)
                .rotationEffect(.degrees(rotation))
                .onAppear {
                    guard !reduceMotion else { return }
                    withAnimation(.linear(duration: 14).repeatForever(autoreverses: false)) {
                        rotation = 360
                    }
                }
        }
        .frame(width: 240, height: 240)
    }

    private var noteGlyph: some View {
        Image(systemName: "music.note")
            .font(.system(size: 64, weight: .heavy, design: .rounded))
            .foregroundColor(.white.opacity(0.9))
            .shadow(color: DesignTokens.ColorToken.accent.opacity(0.5), radius: 12, x: 0, y: 4)
            .scaleEffect(intensity == .active ? 1.08 : 1.0)
            .animation(.spring(response: 0.6, dampingFraction: 0.8), value: intensity)
    }

    private var particles: some View {
        ZStack {
            ForEach(0..<12) { index in
                Circle()
                    .fill(DesignTokens.ColorToken.accent.opacity(0.35))
                    .frame(width: 6, height: 6)
                    .offset(x: CGFloat.random(in: -80...80), y: CGFloat.random(in: -80...80))
                    .opacity(intensity == .active ? 1 : 0.4)
                    .animation(.easeInOut(duration: 2).repeatForever().delay(Double(index) * 0.1), value: intensity)
            }
        }
    }
}
