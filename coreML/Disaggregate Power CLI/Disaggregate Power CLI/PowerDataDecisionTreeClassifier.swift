//
//  PowerDataDecisionTreeClassifier.swift
//  Disaggregate Power CLI
//
//  Created by Reinder Noordmans on 08/06/2024.
//

import Foundation
import TabularData
import CreateML

class PowerDataDecisionTreeClassifier: DPModel {
    var model: MLDecisionTreeClassifier?

    func trainModel(
        targetColumn: String,
        featureColumns: [String],
        maxDepth: Int = 6,
        minLossReduction: Double = 0.0,
        minChildWeight: Double = 0.1
    ) throws {
        let params = MLDecisionTreeClassifier.ModelParameters(
            validation: .dataFrame(
                validationData
            ),
            maxDepth: maxDepth,
            minLossReduction: minLossReduction,
            minChildWeight: minChildWeight,
            randomSeed: Int.random(
                in: 0...1000
            )
        )

        model = try MLDecisionTreeClassifier(
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
        try model.write(to: modelURL)
    }
}
