#include "input_parser.hpp"
#include <fstream>

InputParser::InputParser(std::string inputFile) {
    std::ifstream file(inputFile);
    data = nlohmann::json::parse(file);
    file.close();
}

nlohmann::json InputParser::get_json() {
    return data;
}

Grid InputParser::get_grid() {
    return Grid(data["box"]["n"], data["box"]["size"]);
}

int InputParser::getA() {
    return data["nucleus"]["A"];
}

int InputParser::getZ() {
    return data["nucleus"]["Z"];
}

Calculation InputParser::getCalculation() {
    return Calculation{data["gcg"]["nev"], data["gcg"]["cycles"]};
}