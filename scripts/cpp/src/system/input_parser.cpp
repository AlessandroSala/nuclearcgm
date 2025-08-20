#include "input_parser.hpp"
#include "woods_saxon/deformed_woods_saxon.hpp"
#include <fstream>

InputParser::InputParser(std::string inputFile) {
  std::ifstream file(inputFile);
  data = nlohmann::json::parse(file);
  useCoulomb = data["coulomb"];
  useJ = data["Jterms"];
  std::string interaction = data["interaction"];
  additional = data["additional"];
  pairing = data["pairing"];
  spinOrbit = data["spinOrbit"];
  COMCorr = data["COMCorrection"];

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

WoodsSaxonParameters InputParser::getWS() {
  return WoodsSaxonParameters{
      data["woods_saxon"]["V0"],
      data["woods_saxon"]["r0"],
      data["woods_saxon"]["diffusivity"],
      data["woods_saxon"]["kappa"],
  };
}

WSSpinOrbitParameters InputParser::getWSSO() {
  return WSSpinOrbitParameters{
      data["woods_saxon"]["spin_orbit"]["V0"],
      data["woods_saxon"]["spin_orbit"]["r0"],
      data["woods_saxon"]["diffusivity"],
  };
}

Calculation InputParser::getCalculation() {
  HartreeFock hf = {
      data["hf"]["cycles"], data["hf"]["energyTol"],
      GCGParameters{data["hf"]["gcg"]["nev"], data["hf"]["gcg"]["tol"],
                    data["hf"]["gcg"]["maxIter"], data["hf"]["gcg"]["steps"],
                    data["hf"]["gcg"]["cgTol"]}};
  return Calculation{GCGParameters{data["gcg"]["nev"], data["gcg"]["tol"],
                                   data["gcg"]["maxIter"], data["gcg"]["steps"],
                                   data["gcg"]["cgTol"]},
                     hf};
}
