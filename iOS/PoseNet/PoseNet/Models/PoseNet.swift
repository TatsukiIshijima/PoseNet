//
//  PoseNet.swift
//  PoseNet
//

import Foundation
import UIKit
import FirebaseMLModelInterpreter
import FirebaseMLCommon

class PoseNet {
    
    private let tfliteModelName = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped"
    private let modelName = "posenet_model"
    private var interpreter: ModelInterpreter?
    private var modelElementType: ModelElementType = .float32
    private var inputImage: UIImage?
    
    public static let imageSize = CGSize(width: 257, height: 257)
    public static let bytesPerChannel = 4
    public static let inputChannels = 3
    public static let batchSize = 1
    
    init(inputImage: UIImage) {
        guard let modelPath = Bundle.main.path(forResource: self.tfliteModelName,
                                               ofType: ".tflite") else {
                                                print("Invalid model path.")
                                                return
        }
        let localModel = LocalModel(name: self.modelName, path: modelPath)
        ModelManager.modelManager().register(localModel)
        let options = ModelOptions(remoteModelName: nil,
                                   localModelName: self.modelName)
        self.interpreter = ModelInterpreter.modelInterpreter(options: options)
        
        self.inputImage = inputImage
    }
    
    // シグモイド関数
    private func sigmoid(x: Float) -> Float {
        return (1.0 / (1.0 + exp(-x)))
    }
    
    // 画像を入力データ形式に変換
    private func initInputData(from image: UIImage,
                                 with size: CGSize = PoseNet.imageSize,
                                 componentCount: Int = PoseNet.inputChannels,
                                 batchSize: Int = PoseNet.batchSize
        ) -> ModelInputs? {
        guard let scaledImageData = image.scaledData(
            with: size,
            byteCount: Int(size.width) * Int(size.height) * componentCount * batchSize,
            isQuantized: (modelElementType == .uInt8))
            else {
                print("Failed to get scaled image data with size: \(size).")
                return nil
        }
        let inputs = ModelInputs()
        do {
            try inputs.addInput(scaledImageData)
        } catch let error {
            print("Failed to add input: \(error)")
            return nil
        }
        return inputs
    }
    
    // 出力をDictionaryに変換
    private func initOutput(outputs: ModelOutputs) -> Dictionary<Int, Any>? {
        var outputDictionary = Dictionary<Int, Any>()
        // 1 * 9 * 9 * 17
        let heatMap = try? outputs.output(index: 0) as? [[[[Float]]]]
        guard let _heatMap = heatMap else { return nil }
        print("HeatMap Shape = (\(_heatMap.count),\(_heatMap[0].count),\(_heatMap[0][0].count),\(_heatMap[0][0][0].count))")
        
        // 1 * 9 * 9 * 34
        let offsetMap = try? outputs.output(index: 1) as? [[[[Float]]]]
        guard let _offsetMap = offsetMap else { return nil }
        print("Offset Shape = (\(_offsetMap.count),\(_offsetMap[0].count),\(_offsetMap[0][0].count),\(_offsetMap[0][0][0].count))")
        
        // 1 * 9 * 9 * 32
        let displacementFwdMap = try? outputs.output(index: 2) as? [[[[Float]]]]
        guard let _displacementFwdMap = displacementFwdMap else { return nil }
        print("displecementFwdMap Shape = (\(_displacementFwdMap.count),\(_displacementFwdMap[0].count),\(_displacementFwdMap[0][0].count),\(_displacementFwdMap[0][0][0].count))")
        
        // 1 * 9 * 9 * 32
        let displacementBwdMap = try? outputs.output(index: 3) as? [[[[Float]]]]
        guard let _displacementBwdMap = displacementBwdMap else { return nil }
        print("displacementBwdMap Shape = (\(_displacementBwdMap.count),\(_displacementBwdMap[0].count),\(_displacementBwdMap[0][0].count),\(_displacementBwdMap[0][0][0].count))")
        
        outputDictionary[0] = _heatMap
        outputDictionary[1] = _offsetMap
        outputDictionary[2] = _displacementFwdMap
        outputDictionary[3] = _displacementBwdMap
        return outputDictionary
    }
    
    // 入力形式と出力形式の定義
    private func createInputOutputOptions() -> ModelInputOutputOptions? {
        let ioOptions = ModelInputOutputOptions()
        do {
            try ioOptions.setInputFormat(index: 0,
                                         type: .float32,
                                         dimensions: [1, NSNumber(value: Int( PoseNet.imageSize.height)), NSNumber(value: Int( PoseNet.imageSize.width)), 3])
            try ioOptions.setOutputFormat(index: 0, type: .float32, dimensions: [1, 9, 9, 17])
            try ioOptions.setOutputFormat(index: 1, type: .float32, dimensions: [1, 9, 9, 34])
            try ioOptions.setOutputFormat(index: 2, type: .float32, dimensions: [1, 9, 9, 32])
            try ioOptions.setOutputFormat(index: 3, type: .float32, dimensions: [1, 9, 9, 32])
        } catch let error as NSError {
            print("Failed to set input or output format with error: \(error.localizedDescription)")
            return nil
        }
        return ioOptions
    }
    
    // 姿勢推定
    func estimatePose(completion: @escaping (Person?) -> Void) {
        guard let inputImage = self.inputImage else {
            print("Input image is nil.")
            completion(nil)
            return
        }
        guard let inputs = self.initInputData(from: inputImage) else {
            print("Inputs is nil.")
            completion(nil)
            return
        }
        
        let options = self.createInputOutputOptions()
        guard let _options = options  else {
            print("Options is nil.")
            completion(nil)
            return
        }
        self.interpreter?.run(inputs: inputs, options: _options) { outputs, error in
            guard error == nil, let _outputs = outputs else {
                print("Interpreter run : \(error?.localizedDescription)")
                completion(nil)
                return
            }
            
            let outputDictionary = self.initOutput(outputs: _outputs)
            guard let _outputDictionary = outputDictionary else {
                print("OutputDictionary is nil.")
                completion(nil)
                return
            }
            
            var heatMaps = _outputDictionary[0] as! [[[[Float]]]]
            var offsets = _outputDictionary[1] as! [[[[Float]]]]
            
            let height = heatMaps[0].count
            let width = heatMaps[0][0].count
            let numKeypoints = heatMaps[0][0][0].count
            
            var keyPointPositions = [(Int, Int)](repeating: (0, 0), count: numKeypoints)
            for keyPoint in 0 ..< numKeypoints {
                var maxVal = heatMaps[0][0][0][keyPoint]
                var maxRow = 0
                var maxCol = 0
                for row in 0 ..< height {
                    for col in 0 ..< width {
                        heatMaps[0][row][col][keyPoint] = self.sigmoid(x: heatMaps[0][row][col][keyPoint])
                        if (heatMaps[0][row][col][keyPoint] > maxVal) {
                            maxVal = heatMaps[0][row][col][keyPoint]
                            maxRow = row
                            maxCol = col
                        }
                    }
                }
                keyPointPositions[keyPoint] = (maxRow, maxCol)
            }
            
            var xCoords = [Int](repeating: 0, count: numKeypoints)
            var yCoords = [Int](repeating: 0, count: numKeypoints)
            var confidenceScores = [Float](repeating: 0, count: numKeypoints)
            for (index, position) in keyPointPositions.enumerated() {
                let positionY = keyPointPositions[index].0
                let positionX = keyPointPositions[index].1
                yCoords[index] = Int((Float(position.0) / Float(height - 1) * Float(inputImage.size.height) + offsets[0][positionY][positionX][index]))
                xCoords[index] = Int((Float(position.1) / Float(width - 1) * Float(inputImage.size.width) + offsets[0][positionY][positionX][index + numKeypoints]))
                confidenceScores[index] = heatMaps[0][positionY][positionX][index]
            }
            
            let person = Person()
            //var keyPointList = Array(repeating: KeyPoint(), count: numKeypoints)
            var keyPointList = [KeyPoint]()
            var totalScore: Float = 0.0
            
            for (index, bodyPart) in BodyPart.allCases.enumerated() {
                let keyPoint = KeyPoint()
                keyPoint.bodyPart = bodyPart
                keyPoint.position.x = xCoords[index]
                keyPoint.position.y = yCoords[index]
                keyPoint.score = confidenceScores[index]
                keyPointList.append(keyPoint)
                //keyPointList[index].bodyPart = bodyPart
                //keyPointList[index].position.x = xCoords[index]
                //keyPointList[index].position.y = yCoords[index]
                //keyPointList[index].score = confidenceScores[index]
                //print("confidenceScore[\(index)] : \(confidenceScores[index])")
                
                totalScore += confidenceScores[index]
            }
            
            person.keyPoints = keyPointList
            person.score = totalScore / Float(numKeypoints)
            //print("Total Score : \(person.score)")
            
            completion(person)
        }
    }
}
