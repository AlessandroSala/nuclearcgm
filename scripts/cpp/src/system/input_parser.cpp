#include "input_parser.hpp"
#include "EDF.hpp"
#include "grid.hpp"
#include <fstream>
#include <iostream>

InputParser::InputParser(std::string inputFile) {
  using std::ifstream;
  ifstream file(inputFile);
  data = nlohmann::json::parse(file);
  woodsSaxonData =
      nlohmann::json::parse(ifstream("parameters/woods_saxon.json"));

  hfData = nlohmann::json::parse(
      ifstream("parameters/minimization_parameters.json"));

  _woodsSaxonParameters = WoodsSaxonParameters{
      woodsSaxonData["woods_saxon"]["V0"],
      woodsSaxonData["woods_saxon"]["r0"],
      woodsSaxonData["woods_saxon"]["diffusivity"],
      woodsSaxonData["woods_saxon"]["kappa"],
  };

  _WSSpinOrbitParameters = WSSpinOrbitParameters{
      woodsSaxonData["woods_saxon"]["spin_orbit"]["V0"],
      woodsSaxonData["woods_saxon"]["spin_orbit"]["r0"],
      woodsSaxonData["woods_saxon"]["diffusivity"],
  };

  useCoulomb = data.contains("coulombInteraction")
                   ? data["coulombInteraction"].get<bool>()
                   : false;

  std::string interactionName = data["functional"];
  pairing = data.contains("pairing");
  if (pairing) {
    auto pairingData = data["pairing"];
    std::string pairingType = pairingData["type"];
    if (pairingType == "HFB") {
      pairingType = PairingType::hfb;
    } else {
      pairingType = PairingType::bcs;
    }

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
    pairingType = PairingType::none;
    pairingParameters = PairingParameters{0, 0, 0, 0, 0, 0, 0, false};
  }

  useDIIS = data.contains("useDIIS") ? data["useDIIS"].get<bool>() : true;

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
  nlohmann::json logger =
      nlohmann::json::parse(ifstream("parameters/logger.json"));
  if (logger.contains("log")) {
    for (auto &&entry : logger["log"]) {
      log.push_back(entry);
    }
  }
  nlohmann::json interactionData = nlohmann::json::parse(
      std::ifstream("functionals/" + interactionName + ".json"));

  auto setInteractionOptionalField = [&](std::string field) {
    if (interactionData.contains(field)) {
      return interactionData[field].get<bool>();
    }
    if (data.contains(field)) {
      return data[field].get<bool>();
    }
    return true;
  };

  useJ = setInteractionOptionalField("J2");
  spinOrbit = setInteractionOptionalField("spinOrbit");
  COMCorr = setInteractionOptionalField("COMCorrection");

  interaction = std::make_shared<EDF>(interactionData);

  HartreeFock hf = {data["maxIterations"], data["energyTol"],
                    GCGParameters{hfData["gcg"]["nev"], hfData["gcg"]["tol"],
                                  hfData["gcg"]["maxIter"],
                                  hfData["gcg"]["steps"],
                                  hfData["gcg"]["cgTol"]}};
  calculation =
      Calculation{GCGParameters{hfData["gcg"]["nev"], hfData["gcg"]["tol"],
                                hfData["gcg"]["maxIter"],
                                hfData["gcg"]["steps"], hfData["gcg"]["cgTol"]},
                  hf};

  file.close();
}

nlohmann::json InputParser::getWoodsSaxon() { return data["potentials"][1]; }

std::string InputParser::getOutputName() { return data["outputName"]; }

nlohmann::json InputParser::get_json() { return data; }

Grid InputParser::get_grid() {
  return Grid(data["box"]["axisGridPoints"], data["box"]["sideSize"]);
}

int InputParser::getA() { return A; }

int InputParser::getZ() { return Z; }

WoodsSaxonParameters InputParser::getWS() { return _woodsSaxonParameters; }

WSSpinOrbitParameters InputParser::getWSSO() { return _WSSpinOrbitParameters; }

Calculation InputParser::getCalculation() { return calculation; }
