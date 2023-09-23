//
//  ContentView.swift
//  SwiftFitLine
//
//  Created by Ivan Milinkovic on 23.9.23..
//

import SwiftUI
import Charts

struct Point: Hashable {
    let x: Float
    let y: Float
    init(_ x: Float, _ y: Float) {
        self.x = x
        self.y = y
    }
}

struct Line {
    let k: Float
    let c: Float
    
    init(_ k: Float, _ c: Float) {
        self.k = k
        self.c = c
    }
    
    func f(_ x: Float) -> Float {
        k * x + c
    }
}

func lossf_points(trainingData: [Point], inputs: [Float], isK: Bool) -> [Point] {
    inputs.map { input in
        // todo: is this the correct way to visualise errors on individual dimensions?
        let line = isK ? Line(input, 0) : Line(0, input)
        let ssr = lossf(trainingData: trainingData, line: line)
        return Point(input, ssr)
    }
}

func lossf(trainingData: [Point], line: Line) -> Float {
    var sum = Float(0.0)
    for i in 0..<trainingData.count {
        let pt = trainingData[i]
        sum += lossf(at: pt, line: line)
    }
    return sum
}

func lossf(at pt: Point, line: Line) -> Float {
    let diff = pt.y - line.f(pt.x)
    return diff * diff
}

func gradient_dk(trainingData: [Point], line: Line) -> Float {
    var sum = Float(0.0)
    for i in 0..<trainingData.count {
        let pt = trainingData[i]
        sum += -2 * pt.x * (pt.y - line.k * pt.x - line.c)
    }
    return sum
}

func gradient_dc(trainingData: [Point], line: Line) -> Float {
    var sum = Float(0.0)
    for i in 0..<trainingData.count {
        let pt = trainingData[i]
        sum += -2 * (pt.y - line.k * pt.x - line.c)
    }
    return sum
}

class Solver: ObservableObject {
    
    let trainingData = [Point(1,1), Point(2,2), Point(3,3)]
    var line = Line(0, 0)
    
    var loss_graph_k_points = [Point]()
    var loss_graph_c_points = [Point]()
    
    var current_loss_k_point  = Point(0,0)
    var current_loss_c_point  = Point(0,0)
    var current_loss = Float(0.0)
    
    var learningRate : Float = 0.05
    var prev_grad_k = Float(0)
    
    init() {
        makeLossGraphPoints()
        updateCurrentLossPoints()
    }
    
    func makeLossGraphPoints() {
        let ks = stride(from: Float(-2), through: 3, by: 0.3)
        loss_graph_k_points = lossf_points(trainingData: trainingData, inputs: Array(ks), isK: true)
        
        let cs = stride(from: Float(-2), through: 3, by: 0.3)
        loss_graph_c_points = lossf_points(trainingData: trainingData, inputs: Array(cs), isK: false)
    }
    
    func updateCurrentLossPoints() {
        let current_k_loss = lossf(trainingData: trainingData, line: Line(line.k, 0))
        current_loss_k_point = Point(line.k, current_k_loss)
        
        let current_c_loss = lossf(trainingData: trainingData, line: Line(0, line.c))
        current_loss_c_point = Point(line.c, current_c_loss)
        
        current_loss = lossf(trainingData: trainingData, line: line)
    }
    
    func gradientDescent() {
        let grad_k = gradient_dk(trainingData: trainingData, line: line)
        let step_size_k = grad_k * learningRate
        let new_k = line.k - step_size_k
        
        let grad_c = gradient_dc(trainingData: trainingData, line: line)
        let step_size_c = grad_c * learningRate
        let new_c = line.c - step_size_c
        
        line = Line(new_k, new_c)
        
        updateCurrentLossPoints()
        objectWillChange.send()
    }
    
    private func adjustLearningRate(new_grad_k: Float) {
        if (prev_grad_k == 0.0) {
            prev_grad_k = new_grad_k
            return
        }
        if (prev_grad_k < 0 && new_grad_k > 0) || (prev_grad_k > 0 && new_grad_k < 0) {
            learningRate *= 0.9
        }
    }
}

let solver = Solver()


struct ContentView: View {
    
    @EnvironmentObject var state: Solver
    
    var body: some View {
        VStack {
            Chart {
                ForEach(state.trainingData, id: \.self) { point in
                    PointMark(x: .value("x", point.x), y: .value("y", point.y))
                        .foregroundStyle(.green)
                }
                ForEach(state.trainingData.map(\.x), id: \.self) { x in
                    let y = state.line.f(x)
                    LineMark(x: .value("x", x), y: .value("y", y))
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading)
            }
            
            Spacer(minLength: 30)
            
            VStack {
                Chart {
                    ForEach(state.loss_graph_k_points, id: \.self) { p in
                        PointMark(x: .value("x", p.x), y: .value("y", p.y))
                            .foregroundStyle(.gray)
                    }
                    PointMark(x: .value("x", solver.current_loss_k_point.x),
                              y: .value("y", solver.current_loss_k_point.y))
                        .foregroundStyle(.blue)
                }
                .chartYAxis {
                    AxisMarks(position: .leading)
                }
                Text("k: \(solver.line.k)")
            }
            
            VStack {
                Chart {
                    ForEach(state.loss_graph_c_points, id: \.self) { p in
                        PointMark(x: .value("x", p.x), y: .value("y", p.y))
                            .foregroundStyle(.gray)
                    }
                    PointMark(x: .value("x", solver.current_loss_c_point.x),
                              y: .value("y", solver.current_loss_c_point.y))
                        .foregroundStyle(.blue)
                }
                .chartYAxis {
                    AxisMarks(position: .leading)
                }
                Text("c: \(solver.line.c)")
            }
            
            HStack {
                Button("gradient descent next") {
                    solver.gradientDescent()
                }
                
                Text("loss: \(solver.current_loss)")
                    .font(.title2)
            }
            
            Spacer(minLength: 10)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(solver)
}
