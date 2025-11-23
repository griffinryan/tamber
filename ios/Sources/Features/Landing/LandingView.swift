import SwiftUI

final class LandingViewModel: ObservableObject {
    @Published var prompt: String = ""
    @Published var statusMessage: String = "Idle"
    @Published var isSubmitting = false
    @Published var latestJob: JobStatus?

    private let worker: WorkerClient
    private let env = AppEnvironment.shared

    init(worker: WorkerClient) {
        self.worker = worker
    }

    func submit() {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let parsed = SlashParser(env: env).parse(prompt: prompt)
        Task { @MainActor in
            isSubmitting = true
            statusMessage = "Submitting…"
        }

        Task {
            do {
                let request = JobRequest(
                    prompt: parsed.prompt,
                    model: parsed.model,
                    duration: parsed.duration,
                    cfg: parsed.cfg,
                    seed: parsed.seed,
                    motif: parsed.motif
                )
                let status = try await worker.submitJob(request: request)
                await MainActor.run {
                    self.latestJob = status
                    self.statusMessage = "Queued as \(status.id)"
                    self.isSubmitting = false
                }
            } catch {
                await MainActor.run {
                    self.statusMessage = "Failed: \(error.localizedDescription)"
                    self.isSubmitting = false
                }
            }
        }
    }
}

struct LandingView: View {
    @StateObject private var viewModel = LandingViewModel(worker: WorkerClient(baseURL: AppEnvironment.shared.baseURL))
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            DesignTokens.ColorToken.canvas
                .ignoresSafeArea()
                .overlay(
                    LinearGradient(
                        colors: [
                            DesignTokens.ColorToken.accent.opacity(0.2),
                            Color.clear,
                            DesignTokens.ColorToken.accentWarm.opacity(0.12)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    RadialGradient(
                        colors: [Color.white.opacity(0.08), Color.clear],
                        center: .center,
                        startRadius: 20,
                        endRadius: 320
                    )
                )

            VStack(spacing: 24) {
                Spacer(minLength: 20)

                AnimatedNoteView(intensity: viewModel.isSubmitting ? .active : .idle)
                    .padding(.top, 8)

                Text("Timbre Mobile")
                    .font(DesignTokens.Typography.display(44))
                    .foregroundColor(.white)
                    .shadow(color: Color.black.opacity(0.35), radius: 16, x: 0, y: 10)

                Text("Dreamy glass, CLI-grade control. Compose with slash commands and let the worker sing.")
                    .font(DesignTokens.Typography.rounded(16, weight: .medium))
                    .foregroundColor(Color.white.opacity(0.78))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)

                GlassPanel {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Prompt")
                            .font(DesignTokens.Typography.rounded(14, weight: .semibold))
                            .foregroundColor(Color.white.opacity(0.8))

                        TextField("Add a melody… try /duration 120 /model musicgen-stereo-medium", text: $viewModel.prompt, axis: .vertical)
                            .textFieldStyle(.plain)
                            .foregroundColor(.white)
                            .font(DesignTokens.Typography.rounded(17, weight: .medium))
                            .tint(DesignTokens.ColorToken.accent)
                            .padding(12)
                            .background(
                                RoundedRectangle(cornerRadius: 14)
                                    .fill(Color.white.opacity(0.04))
                            )

                        HStack(spacing: 8) {
                            quickCommandChip("/duration 120")
                            quickCommandChip("/model musicgen-stereo-medium")
                            quickCommandChip("/cfg 6.5")
                        }

                        Button(action: viewModel.submit) {
                            HStack(spacing: 10) {
                                if viewModel.isSubmitting { ProgressView() }
                                Text(viewModel.isSubmitting ? "Sending…" : "Generate")
                                    .font(DesignTokens.Typography.rounded(17, weight: .semibold))
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(AccentButtonStyle())

                        HStack {
                            Circle()
                                .fill(viewModel.isSubmitting ? DesignTokens.ColorToken.accent : Color.green.opacity(0.9))
                                .frame(width: 10, height: 10)
                                .shadow(color: DesignTokens.ColorToken.accent.opacity(0.6), radius: 6)
                            Text(viewModel.statusMessage)
                                .font(DesignTokens.Typography.rounded(14))
                                .foregroundColor(Color.white.opacity(0.85))
                        }
                    }
                }
                .padding(.horizontal, 24)

                Spacer()
            }
            .padding(.bottom, 30)
        }
    }

    private func quickCommandChip(_ value: String) -> some View {
        Button {
            if !viewModel.prompt.contains(value) {
                viewModel.prompt = viewModel.prompt.isEmpty ? value : "\(viewModel.prompt) \(value)"
            }
        } label: {
            Text(value)
                .font(DesignTokens.Typography.rounded(13, weight: .semibold))
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.white.opacity(0.07))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .buttonStyle(.plain)
    }
}

private struct AccentButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.vertical, 14)
            .padding(.horizontal, 16)
            .background(
                LinearGradient(
                    colors: [DesignTokens.ColorToken.accent, DesignTokens.ColorToken.accentWarm],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .opacity(configuration.isPressed ? 0.85 : 1.0)
            .scaleEffect(configuration.isPressed ? 0.99 : 1.0)
            .shadow(color: DesignTokens.ColorToken.accent.opacity(0.45), radius: 18, x: 0, y: 10)
    }
}

private struct SlashParser {
    struct Parsed {
        var prompt: String
        var model: String
        var duration: Int
        var cfg: Double?
        var seed: Int?
        var motif: String?
    }

    let env: AppEnvironment

    func parse(prompt: String) -> Parsed {
        var model = env.defaultModel
        var duration = env.defaultDuration.lowerBound
        var cfg: Double?
        var seed: Int?
        var motif: String?

        var tokens = prompt.components(separatedBy: " ")
        var retained: [String] = []

        while let token = tokens.first {
            tokens.removeFirst()
            switch token {
            case "/duration":
                if let value = tokens.first, let intVal = Int(value) {
                    tokens.removeFirst()
                    duration = max(env.defaultDuration.lowerBound, min(env.defaultDuration.upperBound, intVal))
                }
            case "/model":
                if let value = tokens.first {
                    tokens.removeFirst()
                    model = value
                }
            case "/cfg":
                if let value = tokens.first, let doubleVal = Double(value) {
                    tokens.removeFirst()
                    cfg = doubleVal
                } else if let value = tokens.first, value == "off" {
                    tokens.removeFirst()
                    cfg = nil
                }
            case "/seed":
                if let value = tokens.first, let intVal = Int(value) {
                    tokens.removeFirst()
                    seed = intVal
                }
            case "/motif":
                if let value = tokens.first {
                    tokens.removeFirst()
                    motif = value
                }
            case "/small":
                model = "musicgen-stereo-small"
            case "/medium":
                model = "musicgen-stereo-medium"
            case "/large":
                model = "musicgen-stereo-large"
            default:
                retained.append(token)
            }
        }

        let cleanedPrompt = retained.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        return Parsed(prompt: cleanedPrompt, model: model, duration: duration, cfg: cfg, seed: seed, motif: motif)
    }
}
