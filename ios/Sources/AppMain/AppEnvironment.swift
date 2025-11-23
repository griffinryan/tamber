import Foundation

struct AppEnvironment {
    let baseURL: URL
    let defaultDuration: ClosedRange<Int>
    let defaultModel: String

    static let shared = AppEnvironment()

    init() {
        let envURL = ProcessInfo.processInfo.environment["WORKER_URL"]
        let url = envURL.flatMap { URL(string: $0) } ?? URL(string: "http://localhost:8000")!
        baseURL = url
        defaultDuration = 90...180
        defaultModel = "musicgen-stereo-medium"
    }
}
