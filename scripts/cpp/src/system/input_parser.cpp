#include "input_parser.hpp"
#include <fstream>

InputParser::InputParser(std::string inputFile) {
  std::ifstream file(inputFile);
  data = nlohmann::json::parse(file);
  skyrme = SkyrmeParameters{data["skyrme"]["W0"], data["skyrme"]["t0"],
                            data["skyrme"]["t1"], data["skyrme"]["t2"],
                            data["skyrme"]["t3"], data["skyrme"]["x0"],
                            data["skyrme"]["x1"], data["skyrme"]["x2"],
                            data["skyrme"]["x3"], data["skyrme"]["sigma"]};
  file.close();
}

nlohmann::json InputParser::get_json() { return data; }

Grid InputParser::get_grid() {
  return Grid(data["box"]["n"], data["box"]["size"]);
}

int InputParser::getA() { return data["nucleus"]["A"]; }

int InputParser::getZ() { return data["nucleus"]["Z"]; }

double InputParser::getKappa() { return data["kappa"]; }

Calculation InputParser::getCalculation() {
  HartreeFock hf = {data["hf"]["gcgMaxIter"], data["hf"]["cycles"]};
  return Calculation{data["gcg"]["nev"], data["gcg"]["cycles"], data["gcg"]["tol"], hf};
}
