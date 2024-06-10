//
//  ContentView.swift
//  Disaggregate Power
//
//  Created by Reinder Noordmans on 07/06/2024.
//

import SwiftUI
import CoreML

struct ContentView: View {
    @State private var randomForestProbability: Int64 = 0
    @State private var boostedTreeProbability: Int64 = 0
    @State private var decisionTreeProbability: Int64 = 0
    @State private var logisticRegressionProbability: Int64 = 0
    @State private var wattage: String = "0.0"
    @State private var wattageValue: Double = 0.0
    @State private var date: Date = .now
    @FocusState private var focused: Bool
    private var randomForest: Random_Forest_microwave? = nil
    private var boostedTree: Boosted_Tree_microwave? = nil
    private var decisionTree: Decision_Tree_microwave? = nil
    private var logisticRegression: Logistic_Regression_microwave? = nil

    init() {
        do {
            self.randomForest = try .init(configuration: .init())
            self.boostedTree = try .init(configuration: .init())
            self.decisionTree = try .init(configuration: .init())
            self.logisticRegression = try .init(configuration: .init())
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
                        self.filterWattageInput(newValue)
                        if let wattageDouble = Double(self.wattage) {
                            self.wattageValue = wattageDouble
                        }
                        predict()
                    }

                DatePicker(selection: self.$date, displayedComponents: [.date, .hourAndMinute]) {

                }
            }
            .onChange(of: date) {
                predict()
            }

            Slider(value: $wattageValue, in: 0...3000, step: 1)
                .padding()
                .onChange(of: wattageValue) {
                    self.wattage = String(format: "%.1f", wattageValue)
                    predict()
                }

            VStack {
                PredictionView(modelName: "Random Forest", probability: randomForestProbability)
                PredictionView(modelName: "Boosted Tree", probability: boostedTreeProbability)
                PredictionView(modelName: "Decision Tree", probability: decisionTreeProbability)
                PredictionView(modelName: "Logistic Regression", probability: logisticRegressionProbability)
            }

            Spacer()
        }
        .padding()
    }

    @MainActor
    private func filterWattageInput(_ newValue: String) {
        var filtered = newValue.replacingOccurrences(of: ",", with: ".")
        filtered = filtered.filter { $0.isNumber || $0 == "." }

        if filtered != newValue {
            self.wattage = filtered
        }
    }

    @MainActor
    func predict() {
        do {
            let calendar = Calendar.current
            var components = calendar.dateComponents([.weekday, .hour, .minute], from: date)

            if let weekday = components.weekday {
                components.weekday = (weekday + 5) % 7
            }

            if let weekday = components.weekday, let hour = components.hour, let minute = components.minute {
                print("Predicting with: \(wattageValue), \(weekday), \(hour), \(minute)")
                randomForestProbability = try randomForest?.prediction(input: .init(power_usage: wattageValue, weekday: Int64(weekday), hour: Int64(hour), minute: Int64(minute))).appliance_in_use ?? 0
                boostedTreeProbability = try boostedTree?.prediction(input: .init(power_usage: wattageValue, weekday: Int64(weekday), hour: Int64(hour), minute: Int64(minute))).appliance_in_use ?? 0
                decisionTreeProbability = try decisionTree?.prediction(input: .init(power_usage: wattageValue, weekday: Int64(weekday), hour: Int64(hour), minute: Int64(minute))).appliance_in_use ?? 0
                logisticRegressionProbability = try logisticRegression?.prediction(input: .init(power_usage: wattageValue, weekday: Int64(weekday), hour: Int64(hour), minute: Int64(minute))).appliance_in_use ?? 0
            } else {
                print("Failed to extract date components")
            }
        } catch {
            print(error.localizedDescription)
        }
    }
}

struct PredictionView: View {
    var modelName: String
    var probability: Int64

    var body: some View {
        HStack {
            Text("\(modelName):")
            Spacer()
            Text("\(probability == 1 ? "In Use" : "Not In Use")")
                .foregroundColor(probability == 1 ? .green : .red)
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
