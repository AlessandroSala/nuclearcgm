#include "input_parser.hpp"
#include "EDF.hpp"
#include "grid.hpp"
#include <fstream>
#include <iostream>

bool InputParser::check() {
  if (getA() % 2 != 0 || getZ() % 2 != 0) {
    std::cout << "Nucleus must be even-even" << std::endl;
    return false;
  }
  if (getA() < getZ()) {
    std::cout << "Nucleus A must be larger than Z" << std::endl;
    return false;
  }
  if (pairing) {
    if (pairingN.V0 < 0.0 || pairingP.V0 < 0.0) {
      std::cout << "Pairing strength must be positive" << std::endl;
      return false;
    }
    if (pairingN.eta < 0.0 || pairingP.eta < 0.0 || pairingN.eta > 1.0 ||
        pairingP.eta > 1.0) {
      std::cout << "Pairing eta must be between 0 and 1" << std::endl;
      return false;
    }
    if (pairingN.additionalStates <= 0) {
      std::cout << "Neutron HF basis size must be larger than particle number"
                << std::endl;
      return false;
    }
    if (pairingP.additionalStates <= 0) {
      std::cout << "Proton HF basis size must be larger than particle number"
                << std::endl;
      return false;
    }
    if (pairingN.additionalStates % 2 != 0 ||
        pairingP.additionalStates % 2 != 0) {
      std::cout << "HF basis size must be even" << std::endl;
      return false;
    }
  }
  if (data["box"]["axisMeshPoints"].get<int>() % 2 != 0) {
    std::cout << "Mesh points must be even" << std::endl;
    return false;
  }

  return true;
}

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
                   : true;

  startHFBIter =
      data.contains("startHFBIter") ? data["startHFBIter"].get<int>() : 3;
  constrainCOM =
      data.contains("constrainCOM") ? data["constrainCOM"].get<bool>() : false;
  constraintsTol = data.contains("constraintsTol")
                       ? data["constraintsTol"].get<double>()
                       : 1e-3;
  pairingThreshold = data.contains("pairingThreshold")
                         ? data["pairingThreshold"].get<double>()
                         : 1e-3;

  densityMix =
      data.contains("densityMix") ? data["densityMix"].get<double>() : 0.25;
  A = data["nucleus"]["A"];
  Z = data["nucleus"]["Z"];
  std::string interactionName = data["functional"];
  pairing = data.contains("pairing");
  if (pairing) {
    auto pairingData = data["pairing"];
    std::string pairingTypeStr = pairingData["type"];

    if (pairingTypeStr == "HFB") {
      pairingType = PairingType::hfb;
    } else if (pairingTypeStr == "BCS") {
      pairingType = PairingType::bcs;
    } else {
      pairingType = PairingType::none;
      pairing = false;
    }

    if (pairingData.contains("neutron")) {
      pairingN = PairingParameters{
          pairingData["neutron"].contains("window")
              ? pairingData["neutron"]["window"].get<double>()
              : 5.0,
          pairingData["neutron"]["HFbasisSize"].get<int>() - (A - Z),
          pairingData["neutron"]["V0"],
          pairingData["neutron"].contains("alpha")
              ? pairingData["neutron"]["alpha"].get<double>()
              : 0.0,
          pairingData["neutron"].contains("eta")
              ? pairingData["neutron"]["eta"].get<double>()
              : 0.0,
          pairingData["neutron"].contains("windowBoth")
              ? pairingData["neutron"]["windowBoth"].get<bool>()
              : true,
          true};
      if (!pairingData.contains("proton")) {
        pairingP = PairingParameters{0, 0, 0.0, 0.0, 0.0, false, false};
      }
    }
    if (pairingData.contains("proton")) {
      pairingP = PairingParameters{
          pairingData["proton"].contains("window")
              ? pairingData["proton"]["window"].get<double>()
              : 5.0,
          pairingData["proton"]["HFbasisSize"].get<int>() - Z,
          pairingData["proton"]["V0"],
          pairingData["proton"].contains("alpha")
              ? pairingData["proton"]["alpha"].get<double>()
              : 0.0,
          pairingData["proton"].contains("eta")
              ? pairingData["proton"]["eta"].get<double>()
              : 0.0,
          pairingData["proton"].contains("windowBoth")
              ? pairingData["proton"]["windowBoth"].get<bool>()
              : true,
          true};
      if (!pairingData.contains("neutron")) {
        pairingN = PairingParameters{0, 0, 0.0, 0.0, 0.0, false, false};
      }
    }
    if (!pairingData.contains("neutron") && !pairingData.contains("proton")) {
      int basisSizeN, basisSizeP;
      if (pairingData.contains("HFbasisSizeN")) {
        basisSizeN = pairingData["HFbasisSizeN"].get<int>();
        basisSizeP = pairingData["HFbasisSizeP"].get<int>();
      } else {
        basisSizeN = pairingData["HFbasisSize"];
        basisSizeP = pairingData["HFbasisSize"];
      }
      pairingP = PairingParameters{
          pairingData["window"],
          basisSizeP - Z,
          pairingData["V0"],
          pairingData.contains("alpha") ? pairingData["alpha"].get<double>()
                                        : 0.0,
          pairingData.contains("eta") ? pairingData["eta"].get<double>() : 0.0,
          pairingData.contains("windowBoth")
              ? pairingData["windowBoth"].get<bool>()
              : true,
          true};
      pairingN = PairingParameters{
          pairingData["window"],
          basisSizeN - (A - Z),
          pairingData["V0"],
          pairingData.contains("alpha") ? pairingData["alpha"].get<double>()
                                        : 0.0,
          pairingData.contains("eta") ? pairingData["eta"].get<double>() : 0.0,
          pairingData.contains("windowBoth")
              ? pairingData["windowBoth"].get<bool>()
              : true,
          true};
    }

  } else {
    pairingType = PairingType::none;
    pairingN = PairingParameters{0, 0, 0, 0, 0, false, false};
    pairingP = PairingParameters{0, 0, 0, 0, 0, false, false};
  }

  multipoleConstraints.clear();
  if (data.contains("constraints")) {

    auto constraintsJson =
        data["constraints"].get<std::vector<nlohmann::json>>();
    for (auto &&constraint : constraintsJson) {
      multipoleConstraints.push_back(MultipoleConstraintInput(
          {-1, 10000, constraint["l"], constraint["m"], constraint["target"]}));
    }
  }

  // DIIS still not supported!
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
    initialBeta = data.contains("initialBeta2")
                      ? data["initialBeta2"].get<double>()
                      : 0.0;
  }
  beta3 = data.contains("beta3") ? data["beta3"].get<double>() : 0.0;

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

  auto woodsSaxonGCGParameters = GCGParameters{
      woodsSaxonData["gcg"]["nev"], woodsSaxonData["gcg"]["tol"],
      woodsSaxonData["gcg"]["maxIter"], woodsSaxonData["gcg"]["steps"],
      woodsSaxonData["gcg"]["cgTol"]};

  HartreeFock hf = {data["maxIterations"], data["energyTol"],
                    GCGParameters{hfData["gcg"]["nev"], hfData["gcg"]["tol"],
                                  hfData["gcg"]["maxIter"],
                                  hfData["gcg"]["steps"],
                                  hfData["gcg"]["cgTol"]}};

  calculation = Calculation{woodsSaxonGCGParameters, hf};

  file.close();
}

nlohmann::json InputParser::getWoodsSaxon() { return data["potentials"][1]; }

std::string InputParser::getOutputName() { return data["outputName"]; }

nlohmann::json InputParser::get_json() { return data; }

Grid InputParser::get_grid() {
  return Grid(data["box"]["axisMeshPoints"],
              data["box"]["boxSize"].get<double>() / 2.0);
}

int InputParser::getA() { return A; }

int InputParser::getZ() { return Z; }

WoodsSaxonParameters InputParser::getWS() { return _woodsSaxonParameters; }

WSSpinOrbitParameters InputParser::getWSSO() { return _WSSpinOrbitParameters; }

Calculation InputParser::getCalculation() { return calculation; }
