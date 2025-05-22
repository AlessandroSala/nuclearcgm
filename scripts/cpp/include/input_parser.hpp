
#pragma once 

#include "types.hpp"
#include <string>
#include <vector>
#include <memory>                 // For std::shared_ptr
#include "potential.hpp"
#include "json/json.hpp"
class Grid;

typedef struct {
    int nev;
    int cycles;
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
    Calculation getCalculation();

private:
    nlohmann::json data;
    //Hamiltonian h;


};
