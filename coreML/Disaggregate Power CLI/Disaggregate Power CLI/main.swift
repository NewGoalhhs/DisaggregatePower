//
//  main.swift
//  Disaggregate Power CLI
//
//  Created by Reinder Noordmans on 08/06/2024.
//

import Foundation

func main() {
    // Ask for the input file path
    print("Please enter the path to the training data CSV file:")
    guard let input = readLine(), let trainingDataPath = URL(string: "file://\(input)") else {
        print("Invalid input. Exiting.")
        return
    }

    // Check if the file exists and is readable
    let fileManager = FileManager.default
    if !fileManager.fileExists(atPath: trainingDataPath.path) {
        print("File does not exist at path: \(trainingDataPath.path)")
        return
    }
    if !fileManager.isReadableFile(atPath: trainingDataPath.path) {
        print("File is not readable at path: \(trainingDataPath.path)")
        return
    }

    let targetColumn = "appliance_in_use"
    let featureColumns = ["power_usage", "weekday", "hour", "minute"]

    do {
        let classifier = try PowerDataLogisticRegression(trainingDataPath: trainingDataPath)

        // Train the model
        try classifier.train(targetColumn: targetColumn, featureColumns: featureColumns, maxIterations: 20, l1Penalty: 0.1, l2Penalty: 0.4, stepSize: 0.3, convergenceThreshold: 0.01, featureRescaling: true)

        // Evaluate the model
        let metrics = try classifier.evaluateModel()
        print("Model evaluation metrics: \(metrics)")

        // Ask for the model save path
        print("Please enter the path to save the model:")
        guard let savePath = readLine() else {
            print("Invalid input. Exiting.")
            return
        }

        // Save the model
        try classifier.saveModel(to: savePath)
        print("Model saved successfully to \(savePath)")
    } catch {
        print("An error occurred: \(error)")
    }
}

// Run the main function
main()
