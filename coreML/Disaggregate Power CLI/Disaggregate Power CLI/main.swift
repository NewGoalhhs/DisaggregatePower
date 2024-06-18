//
//  main.swift
//  Disaggregate Power CLI
//
//  Created by Reinder Noordmans on 08/06/2024.
//

import Foundation
import CreateMLComponents

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

    print("Please enter the path to the validation data CSV file:")
    guard let input = readLine(), let validationDataPath = URL(string: "file://\(input)") else {
        print("Invalid input. Exiting.")
        return
    }

    let targetColumn = "appliances_in_use"
    let featureColumns = ["power_usage", "weekday", "hour", "minute"]

    do {
        let classifier = try PowerDataBoostedTreeClassifier(trainingDataPath: trainingDataPath, validationDataPath: validationDataPath)

        // Train the model
        do {
            try classifier.train(targetColumn: targetColumn, featureColumns: featureColumns, maxIterations: 20, maxDepth: 15)
        } catch {
            if let error = error as? OptimizationError {
                print(error.debugDescription)
            }
        }

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
