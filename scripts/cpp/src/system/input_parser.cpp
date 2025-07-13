#include "input_parser.hpp"
#include <fstream>

InputParser::InputParser(std::string inputFile) {
  std::ifstream file(inputFile);
  data = nlohmann::json::parse(file);
  useCoulomb = data["coulomb"];
  std::string interaction = data["interaction"];

  nlohmann::json interactionData = nlohmann::json::parse(
      std::ifstream("interactions/" + interaction + ".json"));

  skyrme = SkyrmeParameters{interactionData["W0"], interactionData["t0"],
                            interactionData["t1"], interactionData["t2"],
                            interactionData["t3"], interactionData["x0"],
                            interactionData["x1"], interactionData["x2"],
                            interactionData["x3"], interactionData["sigma"]};
  file.close();
}

nlohmann::json InputParser::getWoodsSaxon() { return data["potentials"][1]; }

std::string InputParser::getOutputName() { return data["outputName"]; }

nlohmann::json InputParser::get_json() { return data; }

Grid InputParser::get_grid() {
  return Grid(data["box"]["n"], data["box"]["size"]);
}

int InputParser::getA() { return data["nucleus"]["A"]; }

int InputParser::getZ() { return data["nucleus"]["Z"]; }

double InputParser::getKappa() { return data["kappa"]; }

Calculation InputParser::getCalculation() {
  HartreeFock hf = {data["hf"]["cycles"], data["hf"]["energyTol"],
                    GCGParameters{data["hf"]["gcg"]["nev"],
                                  data["hf"]["gcg"]["tol"],
                                  data["hf"]["gcg"]["maxIter"]}};
  return Calculation{data["gcg"]["nev"], data["gcg"]["cycles"],
                     data["gcg"]["tol"], hf};
}
