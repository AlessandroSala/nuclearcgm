
#pragma once 

#include "types.hpp"
#include <string>
#include <vector>
#include <memory>                 // For std::shared_ptr
#include "potential.hpp"

class Grid;

/**
 * @brief Gathers the objects related to a calculation.
 */
class System {
public:
    /**
     */
    System(std::string inputFile);
public:
    Grid grid;
    //Hamiltonian h;


};
