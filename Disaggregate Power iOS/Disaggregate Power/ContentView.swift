//
//  ContentView.swift
//  Disaggregate Power
//
//  Created by Reinder Noordmans on 07/06/2024.
//

import SwiftUI
import CoreML

private enum mlType: String {
    case microwave = "Microwave"
    case multiclass = "Multiclass"
    case torchMulticlass = "Torch Multiclass"
}

struct ContentView: View {
    @State private var modelProbabilities: [String: [Int64: Double]] = [:]
    @State private var multiclassProbability: [String: Double] = [:]
    @State private var wattage: String = "0.0"
    @State private var wattageValue: Double = 0.0
    @State private var date: Date = .now
    @FocusState private var focused: Bool
    @State private var type: mlType = .microwave // Set default type

    private var randomForest: Random_Forest_microwave?
    private var boostedTree: Boosted_Tree_microwave?
    private var decisionTree: Decision_Tree_microwave?
    private var logisticRegression: Logistic_Regression_microwave?

    private var multiclass: dishwasher_microwave_airconditioning?
    private var torch: Torch_dishwasher_microwave_airconditioning?

    init() {
        do {
            randomForest = try Random_Forest_microwave(configuration: .init())
            boostedTree = try Boosted_Tree_microwave(configuration: .init())
            decisionTree = try Decision_Tree_microwave(configuration: .init())
            logisticRegression = try Logistic_Regression_microwave(configuration: .init())
            multiclass = try dishwasher_microwave_airconditioning(configuration: .init())
            torch = try Torch_dishwasher_microwave_airconditioning(configuration: .init())
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

            Picker(selection: $type, label: Text("Model Type")) {
                Text(mlType.microwave.rawValue).tag(mlType.microwave)
                Text(mlType.multiclass.rawValue).tag(mlType.multiclass)
                Text(mlType.torchMulticlass.rawValue).tag(mlType.torchMulticlass)
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding()
            .onChange(of: type) {
                predict()
            }

            Spacer()

            ScrollView {
                VStack {
                    switch type {
                    case .microwave:
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
                    case .multiclass, .torchMulticlass:
                        HighestProbabilityView(probabilities: multiclassProbability)
                    }
                }
            }
        }
        .padding(.horizontal)
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

            switch type {
            case .microwave:
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
            case .multiclass:
                if let mc = multiclass {
                    let prediction = try mc.prediction(input: .init(power_usage: wattageValue, weekday: Int64(weekday), hour: Int64(hour), minute: Int64(minute)))
                    multiclassProbability = prediction.appliances_in_useProbability
                }
            case .torchMulticlass:
                if let tmc = torch {
                    let inputArray = try! MLMultiArray(shape: [1, 3], dataType: .double)
                    inputArray[0] = NSNumber(value: wattageValue)
                    inputArray[1] = NSNumber(value: weekday)
                    inputArray[2] = NSNumber(value: hour)
                    let prediction = try tmc.prediction(input: Torch_dishwasher_microwave_airconditioningInput(input: inputArray))
                    print(prediction.linear_4)
                    print(prediction.linear_4ShapedArray)
                }
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

struct HighestProbabilityView: View {
    var probabilities: [String: Double]

    var body: some View {
        VStack {
            Text("Appliance Usage Probabilities")
                .font(.headline)
                .padding(.bottom)

            ForEach(probabilities.sorted(by: { $0.value > $1.value }), id: \.key) { appliance, probability in
                HStack {
                    Text("\(keyToText(for: appliance)):")
                    Spacer()
                    VStack(alignment: .trailing) {
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
        .padding()
    }

    private func keyToText(for key: String) -> String {
        let appliances = ["Microwave", "Dishwasher", "Airconditioning"]

        let trimmedKey = key.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        let list = trimmedKey.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }

        var activeAppliances = [String]()
        for (index, value) in list.enumerated() {
            if value == "1" {
                activeAppliances.append(appliances[index])
            }
        }

        if activeAppliances.isEmpty {
            return "No appliances are being used."
        } else if activeAppliances.count == 1 {
            return "\(activeAppliances[0])"
        } else {
            let lastAppliance = activeAppliances.removeLast()
            let appliancesString = activeAppliances.joined(separator: ", ") + " and " + lastAppliance
            return "\(appliancesString)"
        }
    }
}

#Preview {
    ContentView()
}
