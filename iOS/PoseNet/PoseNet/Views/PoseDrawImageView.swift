//
//  PoseDrawView.swift
//  PoseNet
//

import Foundation
import UIKit

class PoseDrawImageView: UIImageView {
    
    private let bodyJoints: [(BodyPart, BodyPart)] = [
        (BodyPart.leftWrist,     BodyPart.leftElbow),
        (BodyPart.leftElbow,     BodyPart.leftShoulder),
        (BodyPart.leftShoulder,  BodyPart.rightshoulder),
        (BodyPart.rightshoulder, BodyPart.rightElbow),
        (BodyPart.rightElbow,    BodyPart.rightWrist),
        (BodyPart.leftShoulder,  BodyPart.leftHip),
        (BodyPart.leftHip,       BodyPart.rightHip),
        (BodyPart.rightHip,      BodyPart.rightshoulder),
        (BodyPart.leftHip,       BodyPart.leftKnee),
        (BodyPart.leftKnee,      BodyPart.leftAnkle),
        (BodyPart.rightHip,      BodyPart.rightKnee),
        (BodyPart.rightKnee,     BodyPart.rightAnkle)
    ]
    private let minConfidence: Float = 0.40
    
    var person: Person?
    
    override func draw(_ rect: CGRect) {
        
        guard let person = self.person else { return }
        guard let image = self.image else { return }
        
        UIGraphicsBeginImageContext(image.size)
        
        // 画像を描画
        self.image?.draw(in: CGRect(x: 0, y: 0,
                              width: image.size.width,
                              height: image.size.height))
        
        guard let context = UIGraphicsGetCurrentContext() else { return }
        context.setLineWidth(5.0)
        context.setStrokeColor(UIColor.green.cgColor)
        
        for line in self.bodyJoints {
            if person.keyPoints[line.0.index].score > self.minConfidence && person.keyPoints[line.1.index].score > self.minConfidence {
                // 線の描画
                let startPoint = person.keyPoints[line.0.index].position
                let endPoint = person.keyPoints[line.1.index].position
                context.move(to: CGPoint(x: startPoint.x, y: startPoint.y))
                context.addLine(to: CGPoint(x: endPoint.x, y: endPoint.y))
                context.closePath()
                context.strokePath()
            }
        }
        
        context.setFillColor(UIColor.blue.cgColor)
        
        for keyPoint in person.keyPoints {
            if keyPoint.score > self.minConfidence {
                // 円の描画
                let x = keyPoint.position.x - 7
                let y = keyPoint.position.y - 7
                context.fillEllipse(in: CGRect(x: x, y: y, width: 14, height: 14))
            }
        }
        
        // 描画＆描画領域を解放
        self.image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
    }
}
