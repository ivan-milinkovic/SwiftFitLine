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
        // Identity values: 0 for addition, 1 for multiplication
        let line = isK ? Line(input, 0) : Line(1, input)
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
    
    let trainingData = [Point(1,1), Point(2,2), Point(3,3), Point(4,4)]
    var line = Line(0.2, 0.5)
    
    var loss_graph_k_points = [Point]()
    var loss_graph_c_points = [Point]()
    
    var current_loss_k_point  = Point(0,0)
    var current_loss_c_point  = Point(0,0)
    var current_loss = Float(0.0)
    
    // If the learning rate is too large (depends on training data), the algorithm will not converge
    var learningRate : Float = 0.02
    
    init() {
        makeLossGraphPoints()
        updateCurrentLossPoints()
    }
    
    func makeLossGraphPoints() {
        let ks = stride(from: Float(-2), through: 2, by: 0.2)
        loss_graph_k_points = lossf_points(trainingData: trainingData, inputs: Array(ks), isK: true)
        
        let cs = stride(from: Float(-2), through: 2, by: 0.2)
        loss_graph_c_points = lossf_points(trainingData: trainingData, inputs: Array(cs), isK: false)
    }
    
    func updateCurrentLossPoints() {
        let current_k_loss = lossf(trainingData: trainingData, line: Line(line.k, 0)) // 0 is identity for addition
        current_loss_k_point = Point(line.k, current_k_loss)
        
        let current_c_loss = lossf(trainingData: trainingData, line: Line(1, line.c)) // 1 is identity for multiplication
        current_loss_c_point = Point(line.c, current_c_loss)
        
        current_loss = lossf(trainingData: trainingData, line: line)
    }
    
    func gradientDescentStep(updateObservers: Bool) {
        
        let grad_k = gradient_dk(trainingData: trainingData, line: line)
        let step_size_k = grad_k * learningRate
        let new_k = line.k - step_size_k
        
        let grad_c = gradient_dc(trainingData: trainingData, line: line)
        let step_size_c = grad_c * learningRate
        let new_c = line.c - step_size_c
        
        line = Line(new_k, new_c)
        
        updateCurrentLossPoints()
        
        if updateObservers {
            objectWillChange.send()
        }
    }
    
    func gradientDescentAuto() {
        
        let max_iter = 1000
        let loss_target : Float = 0.0005
        var i = 0
        while true {
            if current_loss < loss_target {
                print("auto: loss target achieved")
                break
            }
            if i > max_iter {
                print("auto: max iterations")
                break
            }
            gradientDescentStep(updateObservers: false)
            i += 1
        }
        objectWillChange.send()
    }
    
    var animationTimer : Timer?
    
    func gradientDescentAutoAnimate() {
        let max_iter = 1000
        let loss_target : Float = 0.001
        var i = 0
        
        animationTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { timer in
            if self.current_loss < loss_target {
                print("auto: loss target achieved")
                self.stopAnimating()
            }
            if i > max_iter {
                print("auto: max iterations")
                self.stopAnimating()
            }
            self.gradientDescentStep(updateObservers: true)
            i += 1
        }
    }
    
    var isAnimating: Bool {
        animationTimer != nil
    }
    
    func stopAnimating() {
        animationTimer?.invalidate()
        animationTimer = nil
        objectWillChange.send()
    }
}

let solver = Solver()


struct ContentView: View {
    
    @EnvironmentObject var solver: Solver
    
    var body: some View {
        VStack {
            Chart {
                ForEach(solver.trainingData, id: \.self) { point in
                    PointMark(x: .value("x", point.x), y: .value("y", point.y))
                        .foregroundStyle(.green)
                }
                ForEach(solver.trainingData.map(\.x), id: \.self) { x in
                    let y = solver.line.f(x)
                    LineMark(x: .value("x", x), y: .value("y", y))
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading)
            }
            
            Spacer(minLength: 30)
            
            VStack {
                Chart {
                    ForEach(solver.loss_graph_k_points, id: \.self) { p in
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
                    ForEach(solver.loss_graph_c_points, id: \.self) { p in
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
                Text("Gradient Descent")
                Button("Auto") {
                    if solver.isAnimating { solver.stopAnimating() }
                    solver.gradientDescentAuto()
                }
                Button(solver.isAnimating ? "Stop" : "Animate") {
                    if solver.isAnimating {
                        solver.stopAnimating()
                    } else {
                        solver.gradientDescentAutoAnimate()
                    }
                }
                Button("Next") {
                    if solver.isAnimating { solver.stopAnimating() }
                    solver.gradientDescentStep(updateObservers: true)
                }
                Text("loss: \(solver.current_loss)")
                    .font(.title2)
            }
            
            Spacer(minLength: 10)
        }
        .padding()
    }
}

#Preview {
    ContentView()
        .environmentObject(solver)
}
