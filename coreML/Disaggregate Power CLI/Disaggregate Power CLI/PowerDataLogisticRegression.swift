//
//  PowerDataLogisticRegression.swift
//  Disaggregate Power CLI
//
//  Created by Reinder Noordmans on 08/06/2024.
//

import Foundation
import TabularData
import CreateML

class PowerDataLogisticRegression: DPModel {
    var model: MLLogisticRegressionClassifier?

    func train(
        targetColumn: String,
        featureColumns: [String],
        maxIterations: Int = 10,
        l1Penalty: Double = 0,
        l2Penalty: Double = 0.01,
        stepSize: Double = 1,
        convergenceThreshold: Double = 0.01,
        featureRescaling: Bool = false
    ) throws {
        let params = MLLogisticRegressionClassifier.ModelParameters(validation: .dataFrame(validationData), maxIterations: maxIterations, l1Penalty: l1Penalty, l2Penalty: l2Penalty, stepSize: stepSize, convergenceThreshold: convergenceThreshold, featureRescaling: featureRescaling)

        model = try MLLogisticRegressionClassifier(
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
