//
//  DPModel.swift
//  Disaggregate Power CLI
//
//  Created by Reinder Noordmans on 08/06/2024.
//

import Foundation
import CreateML
import CreateMLComponents
import TabularData

class DPModel {
    var metadata: MLModelMetadata

    let trainingData: DataFrame
    let validationData: DataFrame

    init(trainingDataPath: URL, validationDataPath: URL? = nil) throws {
        self.metadata = .init()
        self.metadata.author = "Reinder Noordans"
        self.metadata.additional = [
            "com.apple.coreml.model.preview.type": "tabularClassifier"
        ]

        let tdata = try Data(contentsOf: trainingDataPath)
        let trainingData = try DataFrame(csvData: tdata)
        if let validationDataPath {
            self.trainingData = trainingData
            let vdata = try Data(contentsOf: validationDataPath)
            self.validationData = try DataFrame(csvData: vdata)
        } else {
            let (trainData, valData) = trainingData.randomSplit(by: 0.8, seed: 42)
            self.trainingData = trainData.base
            self.validationData = valData.base
        }
    }
}
