
#pragma once
#include "potential.hpp"
#include "spin_orbit/deformed_spin_orbit.hpp"
#include "types.hpp"
#include "woods_saxon/deformed_woods_saxon.hpp"
#include "json/json.hpp"
#include <memory> // For std::shared_ptr
#include <string>
#include <vector>
class Grid;

typedef struct GCGParameters {
  int nev;
  double tol;
  int maxIter;
} CGCParameters;
typedef struct {
  double W0;
  double t0;
  double t1;
  double t2;
  double t3;
  double x0;
  double x1;
  double x2;
  double x3;
  double sigma;
} SkyrmeParameters;
typedef struct HartreeFock {
  int cycles;
  GCGParameters gcg;
} HartreeFock;
typedef struct {
  int nev;
  int cycles;
  double tol;
  HartreeFock hf;
} Calculation;

/**
 * @brief Gathers the objects related to a calculation.
 */
class InputParser {
public:
  /**
   * @param inputFile The path to the input file in the input directory.
   */
  InputParser(std::string inputFile);
  nlohmann::json get_json();
  Grid get_grid();
  int getA();
  int getZ();
  double getKappa();
  Calculation getCalculation();
  std::string getOutputName();
  nlohmann::json getWoodsSaxon();

  SkyrmeParameters skyrme;

private:
  nlohmann::json data;
  // Hamiltonian h;
};
