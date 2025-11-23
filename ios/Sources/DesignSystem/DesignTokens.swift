import SwiftUI

enum DesignTokens {
    enum ColorToken {
        static let glassBase = Color.white.opacity(0.04)
        static let glassStroke = Color.white.opacity(0.3)
        static let accent = Color(red: 0.35, green: 0.82, blue: 0.96)
        static let accentWarm = Color(red: 0.99, green: 0.76, blue: 0.41)
        static let success = Color(red: 0.48, green: 0.86, blue: 0.52)
        static let warning = Color(red: 1.0, green: 0.72, blue: 0.35)
        static let error = Color(red: 0.96, green: 0.36, blue: 0.41)
        static let canvas = LinearGradient(
            colors: [
                Color(red: 0.06, green: 0.09, blue: 0.16),
                Color(red: 0.04, green: 0.06, blue: 0.11)
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    enum Typography {
        static func display(_ size: CGFloat) -> Font {
            Font.custom("Alagard", size: size, relativeTo: .largeTitle)
        }

        static func rounded(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
            Font.system(size: size, weight: weight, design: .rounded)
        }
    }
}
