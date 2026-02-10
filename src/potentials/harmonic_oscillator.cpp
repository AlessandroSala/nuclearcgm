#include <cmath>
#include "harmonic_oscillator.hpp"
#include "constants.hpp"

HarmonicOscillatorPotential::HarmonicOscillatorPotential(double omega_x_, double omega_y_, double omega_z_)
    : omega_x(omega_x_), omega_y(omega_y_), omega_z(omega_z_) {}
double HarmonicOscillatorPotential::getValue(double x, double y, double z) const {
    return nuclearConstants::m*(0.5*omega_x*omega_x*x*x + 0.5*omega_y*omega_y*y*y + 0.5*omega_z*omega_z*z*z);
}