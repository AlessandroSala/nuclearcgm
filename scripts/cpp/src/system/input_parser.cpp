#include "input_parser.hpp"
#include "grid.hpp"
#include <fstream>
#include <iostream>

InputParser::InputParser(std::string inputFile) {
  std::ifstream file(inputFile);
  data = nlohmann::json::parse(file);

  useCoulomb = data["coulomb"];
  useJ = data["Jterms"];
  std::string interaction = data["interaction"];
  pairing = data.contains("pairing");
  if (pairing) {
    auto pairingData = data["pairing"];
    if (pairingData.contains("neutron")) {
      pairingParameters = PairingParameters{
          pairingData["window"],
          pairingData["neutron"]["additionalStates"],
          pairingData["proton"]["additionalStates"],
          pairingData["neutron"]["V0"],
          pairingData["proton"]["V0"],
          pairingData.contains("alpha") ? pairingData["alpha"].get<double>()
                                        : 0.0,
          pairingData.contains("eta") ? pairingData["eta"].get<double>() : 0.0,
          pairingData.contains("windowBoth")
              ? pairingData["windowBoth"].get<bool>()
              : false,
      };
    } else {
      pairingParameters = PairingParameters{
          pairingData["window"],
          pairingData["additionalStates"],
          pairingData["additionalStates"],
          pairingData["V0"],
          pairingData["V0"],
          pairingData.contains("alpha") ? pairingData["alpha"].get<double>()
                                        : 0.0,
          pairingData.contains("eta") ? pairingData["eta"].get<double>() : 0.0,
          pairingData.contains("windowBoth")
              ? pairingData["windowBoth"].get<bool>()
              : false,
      };
    }

  } else {
    pairingParameters = PairingParameters{0, 0, 0, 0, 0, 0, 0, false};
  }

  useDIIS = data.contains("useDIIS") ? data["useDIIS"].get<bool>() : true;

  spinOrbit = data["spinOrbit"];
  COMCorr = data["COMCorrection"];
  outputDirectory = data["outputDirectory"];
  calculationType = data.contains("deformation")
                        ? CalculationType::deformation_curve
                        : CalculationType::ground_state;
  if (calculationType == CalculationType::deformation_curve) {
    deformationCurve = DeformationCurve{data["deformation"]["start"],
                                        data["deformation"]["end"],
                                        data["deformation"]["step"]};
    initialBeta = data["deformation"].contains("guess")
                      ? data["deformation"]["guess"].get<double>()
                      : deformationCurve.start * 1;
  } else {
    initialBeta =
        data.contains("initialBeta") ? data["initialBeta"].get<double>() : 0.0;
  }
  beta3 = data.contains("beta3") ? data["beta3"].get<double>() : 0.0;

  A = data["nucleus"]["A"];
  Z = data["nucleus"]["Z"];

  log.clear();
  if (data.contains("log")) {
    for (auto &&entry : data["log"]) {
      log.push_back(entry);
    }
  }
  nlohmann::json interactionData = nlohmann::json::parse(
      std::ifstream("interactions/" + interaction + ".json"));

  skyrme = SkyrmeParameters{interactionData["W0"], interactionData["t0"],
                            interactionData["t1"], interactionData["t2"],
                            interactionData["t3"], interactionData["x0"],
                            interactionData["x1"], interactionData["x2"],
                            interactionData["x3"], interactionData["sigma"]};
  std::cout << "Interaction: " << interaction << std::endl;
  std::cout << "t0: " << skyrme.t0 << ", t1: " << skyrme.t1
            << ", t2: " << skyrme.t2 << ", t3: " << skyrme.t3 << std::endl;
  std::cout << "W0: " << skyrme.W0 << ", x0: " << skyrme.x0
            << ", x1: " << skyrme.x1 << ", x2: " << skyrme.x2
            << ", x3: " << skyrme.x3 << std::endl;
  std::cout << "sigma: " << skyrme.sigma << std::endl;
  file.close();
}

nlohmann::json InputParser::getWoodsSaxon() { return data["potentials"][1]; }

std::string InputParser::getOutputName() { return data["outputName"]; }

nlohmann::json InputParser::get_json() { return data; }

Grid InputParser::get_grid() {
  return Grid(data["box"]["n"], data["box"]["size"]);
}

int InputParser::getA() { return A; }

int InputParser::getZ() { return Z; }

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
