//
//  ContentView.swift
//  Disaggregate Power
//
//  Created by Reinder Noordmans on 07/06/2024.
//

import SwiftUI
import CoreML

struct ContentView: View {
    @State private var modelProbabilities: [String: [Int64: Double]] = [:]
    @State private var wattage: String = "0.0"
    @State private var wattageValue: Double = 0.0
    @State private var date: Date = .now
    @FocusState private var focused: Bool

    private var randomForest: Random_Forest_microwave?
    private var boostedTree: Boosted_Tree_microwave?
    private var decisionTree: Decision_Tree_microwave?
    private var logisticRegression: Logistic_Regression_microwave?

    init() {
        do {
            randomForest = try Random_Forest_microwave(configuration: .init())
            boostedTree = try Boosted_Tree_microwave(configuration: .init())
            decisionTree = try Decision_Tree_microwave(configuration: .init())
            logisticRegression = try Logistic_Regression_microwave(configuration: .init())
        } catch {
            print(error.localizedDescription)
        }
    }

    var body: some View {
        VStack {
            HStack {
                TextField("Wattage: ", text: $wattage)
                    .keyboardType(.decimalPad)
                    .focused($focused)
                    .onChange(of: wattage) { _, newValue in
                        filterWattageInput(newValue)
                        if let wattageDouble = Double(wattage) {
                            wattageValue = wattageDouble
                        }
                        predict()
                    }

                DatePicker(selection: $date, displayedComponents: [.date, .hourAndMinute]) {}
            }
            .onChange(of: date) {
                predict()
            }

            Slider(value: $wattageValue, in: 0...3000, step: 1)
                .padding()
                .onChange(of: wattageValue) {
                    wattage = String(wattageValue)
                    predict()
                }

            Spacer()

            VStack {
                if let rfProbability = modelProbabilities["Random Forest"]?[1] {
                    PredictionView(modelName: "Random Forest", probability: rfProbability)
                }
                if let btProbability = modelProbabilities["Boosted Tree"]?[1] {
                    PredictionView(modelName: "Boosted Tree", probability: btProbability)
                }
                if let dtProbability = modelProbabilities["Decision Tree"]?[1] {
                    PredictionView(modelName: "Decision Tree", probability: dtProbability)
                }
                if let lrProbability = modelProbabilities["Logistic Regression"]?[1] {
                    PredictionView(modelName: "Logistic Regression", probability: lrProbability)
                }
            }
        }
        .padding()
    }

    @MainActor
    private func filterWattageInput(_ newValue: String) {
        var filtered = newValue.replacingOccurrences(of: ",", with: ".")
        filtered = filtered.filter { $0.isNumber || $0 == "." }

        if filtered != newValue {
            wattage = filtered
        }
    }

    @MainActor
    private func predict() {
        do {
            let calendar = Calendar.current
            var components = calendar.dateComponents([.weekday, .hour, .minute], from: date)

            if let weekday = components.weekday {
                components.weekday = (weekday + 5) % 7
            }

            guard let weekday = components.weekday,
                  let hour = components.hour,
                  let minute = components.minute else {
                print("Failed to extract date components")
                return
            }

            if let rf = randomForest {
                let prediction = try rf.prediction(input: .init(power_usage: wattageValue, weekday: Int64(weekday), hour: Int64(hour), minute: Int64(minute)))
                modelProbabilities["Random Forest"] = prediction.appliance_in_useProbability
            }

            if let bt = boostedTree {
                let prediction = try bt.prediction(input: .init(power_usage: wattageValue, weekday: Int64(weekday), hour: Int64(hour), minute: Int64(minute)))
                modelProbabilities["Boosted Tree"] = prediction.appliance_in_useProbability
            }

            if let dt = decisionTree {
                let prediction = try dt.prediction(input: .init(power_usage: wattageValue, weekday: Int64(weekday), hour: Int64(hour), minute: Int64(minute)))
                modelProbabilities["Decision Tree"] = prediction.appliance_in_useProbability
            }

            if let lr = logisticRegression {
                let prediction = try lr.prediction(input: .init(power_usage: wattageValue, weekday: Int64(weekday), hour: Int64(hour), minute: Int64(minute)))
                modelProbabilities["Logistic Regression"] = prediction.appliance_in_useProbability
            }
        } catch {
            print(error.localizedDescription)
        }
    }
}

struct PredictionView: View {
    var modelName: String
    var probability: Double

    var body: some View {
        HStack {
            Text("\(modelName):")
            Spacer()
            VStack(alignment: .trailing) {
                Text("\(probability >= 0.5 ? "In Use" : "Not In Use")")
                    .font(.headline)
                    .foregroundColor(probability >= 0.5 ? .green : .red)

                Text(probability.formatted(.percent))
                    .font(.footnote)
                    .foregroundColor(.gray)
            }
        }
        .padding()
        .background(Color.gray.opacity(0.2))
        .cornerRadius(10)
        .padding(.horizontal)
    }
}

#Preview {
    ContentView()
}
