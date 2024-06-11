//
//  PowerDataBoostedTreeClassifier.swift
//  Disaggregate Power CLI
//
//  Created by Reinder Noordmans on 08/06/2024.
//

import Foundation
import TabularData
import CreateML

class PowerDataBoostedTreeClassifier: DPModel {
    var model: MLBoostedTreeClassifier?

    func trainModel(
        targetColumn: String,
        featureColumns: [String],
        maxIterations: Int = 10,
        maxDepth: Int = 6,
        minLossReduction: Double = 0.0,
        minChildWeight: Double = 0.1,
        rowSubsampleRatio: Double = 1.0,
        columnSubsampleRatio: Double = 1.0,
        stepSize: Double = 0.3
    ) throws {
        let params = MLBoostedTreeClassifier.ModelParameters(
            validation: .dataFrame(
                validationData
            ),
            maxDepth: maxDepth,
            maxIterations: maxIterations,
            minLossReduction: minLossReduction,
            minChildWeight: minChildWeight,
            randomSeed: Int.random(
                in: 0...1000
            ),
            stepSize: stepSize,
            rowSubsample: rowSubsampleRatio,
            columnSubsample: columnSubsampleRatio
        )
        
        model = try .init(
            trainingData: trainingData,
            targetColumn: targetColumn,
            featureColumns: featureColumns,
            parameters: params
        )
    }

    func evaluateModel() throws -> MLClassifierMetrics {
        guard let model = model else {
            throw NSError(domain: "Model not trained or validation data not available", code: 1, userInfo: nil)
        }
        let evaluationMetrics = model.evaluation(on: validationData)
        return evaluationMetrics
    }

    func saveModel(to path: String) throws {
        guard let model = model else {
            throw NSError(domain: "Model not trained", code: 1, userInfo: nil)
        }
        let modelURL = URL(fileURLWithPath: path)
        try model.write(to: modelURL, metadata: metadata)
    }
}
