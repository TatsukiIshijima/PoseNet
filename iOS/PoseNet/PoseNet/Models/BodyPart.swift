//
//  BodyPart.swift
//  PoseNet
//

enum BodyPart: CaseIterable {
    case nose
    case leftEye
    case rightEye
    case leftEar
    case rightEar
    case leftShoulder
    case rightshoulder
    case leftElbow
    case rightElbow
    case leftWrist
    case rightWrist
    case leftHip
    case rightHip
    case leftKnee
    case rightKnee
    case leftAnkle
    case rightAnkle
}

extension CaseIterable where Self: Equatable {
    var index: Int {
        guard let index = Self.allCases.firstIndex(of: self) as? Int else { fatalError() }
        return index
    }
}
