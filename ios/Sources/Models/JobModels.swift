import Foundation

struct JobRequest: Codable {
    var prompt: String
    var model: String
    var duration: Int
    var cfg: Double?
    var seed: Int?
    var motif: String?
}

struct JobStatus: Codable, Identifiable {
    enum State: String, Codable {
        case queued
        case running
        case completed
        case failed
    }

    var id: String
    var state: State
    var message: String?
}

struct Artifact: Codable {
    var jobId: String
    var audioURL: URL
    var metadataURL: URL
}
