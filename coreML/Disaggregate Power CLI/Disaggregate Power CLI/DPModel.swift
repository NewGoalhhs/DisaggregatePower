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
    let trainingData: DataFrame
    let validationData: DataFrame

    init(trainingDataPath: URL) throws {
        let data = try Data(contentsOf: trainingDataPath)
        let csvData = try DataFrame(csvData: data)
        let (trainData, valData) = csvData.randomSplit(by: 0.8, seed: 42)
        self.trainingData = trainData.base
        self.validationData = valData.base
    }
}
