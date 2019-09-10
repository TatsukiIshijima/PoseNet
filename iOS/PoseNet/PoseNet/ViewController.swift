//
//  ViewController.swift
//  PoseNet
//

import UIKit

class ViewController: UIViewController {

    private let sampleImage = UIImage(named: "sample")
    
    @IBOutlet weak var poseDrawImageView: PoseDrawImageView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        poseDrawImageView.image = sampleImage
        guard let inputImage = self.sampleImage else { return }
        let poseNet: PoseNet = PoseNet(inputImage: inputImage)
        poseNet.estimatePose(completion: { person in
            guard let person = person else { return }
            print("Total Score : \(person.score)")
            self.poseDrawImageView.person = person
            self.poseDrawImageView.draw(CGRect(x: 0, y: 0,
                                        width: self.poseDrawImageView.bounds.width,
                                        height: self.poseDrawImageView.bounds.height))
        })
    }


}

