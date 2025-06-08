
#pragma once
#include "woods_saxon/deformed_woods_saxon.hpp"
#include "spin_orbit/deformed_spin_orbit.hpp"
#include "potential.hpp"
#include "types.hpp"
#include "json/json.hpp"
#include <memory> // For std::shared_ptr
#include <string>
#include <vector>
class Grid;

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
typedef struct {
  int nev;
  int cycles;
  int hfCycles; // eigenpair = gcgm_complex_no_B(ham.build_matrix5p(),
                // harmonic_oscillator_guess(grid, calc.nev, grid.get_a()),
                // calc.nev, 35 + 0.01, calc.cycles, 1.0e-3, 20, 1.0e-4 /
                // (calc.nev), false, 1);
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

  SkyrmeParameters skyrme;


private:
  nlohmann::json data;
  // Hamiltonian h;
};
