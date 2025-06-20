#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "NeuralNetwork.h"

int main() {
    std::vector<std::vector<double>> X, y;

    std::ifstream file("numericalOptimization_Data.csv");
    std::string line;

    int lineNumber = 0;

    while (std::getline(file, line)) {
        lineNumber++;
        if (lineNumber < 5) continue; 
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {
            try {
                if (!cell.empty()) row.push_back(std::stod(cell));
            }
        }

        if (row.size() < 579) {
            std::cout << "? 경고: " << lineNumber << "행 누락됨 (컬럼 수 = " << row.size() << ")" << std::endl;
            continue;
        }

        std::vector<double> input(row.begin(), row.begin() + 576);
        std::vector<double> label(row.begin() + 576, row.begin() + 579);

        X.push_back(input);
        y.push_back(label);
    }


    std::cout << "데이터 로딩 완료! 샘플 개수: " << X.size() << std::endl;

    // 신경망 초기화
    NeuralNetwork nn(576, 3, 2, 32, 0.1);
    nn.train(X, y, 50);  // epoch 수는 예시로 50

    // 예측 출력
    std::vector<double> result = nn.forward(X[0]);
    std::cout << "첫 번째 입력의 예측 결과: ";
    for (double val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
