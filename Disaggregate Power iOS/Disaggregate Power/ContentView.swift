//
//  ContentView.swift
//  Disaggregate Power
//
//  Created by Reinder Noordmans on 07/06/2024.
//

import SwiftUI
import CoreML

struct ContentView: View {
    @State private var probability: Int64 = 0
    @State private var wattage: String = "0.0"
    @State private var date: Date = .now
    @FocusState private var focused: Bool
    private var model2: Random_Forest_microwave_copy? = nil

    init() {
        do {
            self.model2 = try .init(configuration: .init())
        } catch {
            print(error.localizedDescription)
        }
    }

    var body: some View {
        ZStack {
            Group {
                if probability == 0 {
                    Color.red
                } else {
                    Color.green
                }
            }.ignoresSafeArea()

            VStack {
                HStack {
                    TextField("Wattage: ", text: $wattage)
                        .keyboardType(.decimalPad)
                        .focused($focused)

                    DatePicker(selection: self.$date, displayedComponents: [.date, .hourAndMinute]) {

                    }
                }
                .onChange(of: wattage) { _, newValue in
                    self.filterWattageInput(newValue)
                    predict()
                }
                .onChange(of: date) {
                    predict()
                }
                Spacer()
            }
            .padding()
        }
//        .onTapGesture {
//            self.focused = false
//        }
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
            let components = calendar.dateComponents([.weekday, .hour, .minute], from: date)

            if let weekday = components.weekday, let hour = components.hour, let minute = components.minute {
                print("Predicting with: \(Double(self.wattage) ?? 0), \(weekday), \(hour), \(minute)")
                probability = try model2?.prediction(input: .init(power_usage: Double(self.wattage) ?? 0.0, weekday: Int64(weekday), hour: Int64(hour), minute: Int64(minute))).appliance_in_use ?? 0
            } else {
                print("Failed to extract date components")
            }
        } catch {
            print(error.localizedDescription)
        }
    }
}

#Preview {
    ContentView()
}
