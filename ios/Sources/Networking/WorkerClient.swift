import Foundation

actor WorkerClient {
    private let session: URLSession
    private let baseURL: URL

    init(baseURL: URL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
    }

    func submitJob(request: JobRequest) async throws -> JobStatus {
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("generate"))
        urlRequest.httpMethod = "POST"
        urlRequest.addValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)

        let (data, response) = try await session.data(for: urlRequest)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }

    func fetchStatus(for id: String) async throws -> JobStatus {
        let url = baseURL.appendingPathComponent("status/\(id)")
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }

    func fetchArtifact(for id: String) async throws -> Artifact {
        let url = baseURL.appendingPathComponent("artifact/\(id)")
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(Artifact.self, from: data)
    }
}
