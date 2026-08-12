// ws0_vision_ocr.swift — batch OCR for answer-leak detection (WS0)
// Usage: ws0_vision_ocr <file-with-image-paths, one per line>
// Output: JSONL to stdout: {"path": "...", "ok": true, "text": "..."} per image.
// Uses macOS Vision framework, zh-Hans + en, accurate mode. No network, no cost.

import Foundation
import Vision
import CoreImage

func ocr(path: String) -> [String: Any] {
    let url = URL(fileURLWithPath: path)
    guard let ciImage = CIImage(contentsOf: url) else {
        return ["path": path, "ok": false, "error": "cannot_load_image"]
    }
    let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["zh-Hans", "en-US"]
    request.usesLanguageCorrection = false
    do {
        try handler.perform([request])
    } catch {
        return ["path": path, "ok": false, "error": "vision_error: \(error.localizedDescription)"]
    }
    let lines: [String] = (request.results ?? []).compactMap { obs in
        obs.topCandidates(1).first?.string
    }
    return ["path": path, "ok": true, "text": lines.joined(separator: "\n")]
}

guard CommandLine.arguments.count >= 2,
      let listData = try? String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8) else {
    FileHandle.standardError.write("usage: ws0_vision_ocr <path-list-file>\n".data(using: .utf8)!)
    exit(2)
}

let paths = listData.split(separator: "\n").map(String.init).filter { !$0.isEmpty }
let out = FileHandle.standardOutput
for p in paths {
    let result = ocr(path: p)
    if let data = try? JSONSerialization.data(withJSONObject: result),
       let line = String(data: data, encoding: .utf8) {
        out.write((line + "\n").data(using: .utf8)!)
    }
}
