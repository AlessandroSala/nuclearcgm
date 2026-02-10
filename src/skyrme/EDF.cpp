#include "EDF.hpp"
#include <iostream>

EDF::EDF(nlohmann::json functional) {
  std::cout << functional << std::endl;
  if (functional.contains("t0")) {
    std::cout << "Converting interaction to EDF" << std::endl;

    double W0, W0_pr;
    W0 = functional["W0"];
    if (functional.contains("W0_pr")) {
      W0_pr = functional["W0_pr"];
    } else {
      W0_pr = W0;
    }
    double te = 0.0;
    double to = 0.0;
    if (functional.contains("te")) {
      te = functional["te"];
      to = functional["to"];
    } else if (functional.contains("aT")) {
      to = 4 * functional["aT"].get<double>() / 5;
      te = 8 * functional["bT"].get<double>() / 5 - to;
    }

    SkyrmeParameters interaction = {
        .W0 = W0,
        .W0_pr = W0_pr,
        .t0 = functional["t0"],
        .t1 = functional["t1"],
        .t2 = functional["t2"],
        .t3 = functional["t3"],
        .x0 = functional["x0"],
        .x1 = functional["x1"],
        .x2 = functional["x2"],
        .x3 = functional["x3"],
        .sigma = functional["sigma"],
        .te = te,
        .to = to,
    };

    setFromInteraction(interaction);

    std::cout << "C0rr: " << params.C0rr << ", C1rr: " << params.C1rr
              << ", C0Drr: " << params.C0Drr << ", C1Drr: " << params.C1Drr
              << ", C0rt: " << params.C0rt << ", C1rt: " << params.C1rt
              << ", C0J2: " << params.C0J2 << ", C1J2: " << params.C1J2
              << ", C0rdr: " << params.C0rdr << ", C1rdr: " << params.C1rdr
              << ", C0nJ: " << params.C0nJ << ", C1nJ: " << params.C1nJ
              << ", sigma: " << params.sigma << std::endl;
  } else {
    params = (FunctionalParameters{
        functional["C0rr"], functional["C1rr"], functional["C0Drr"],
        functional["C1Drr"], functional["C0rt"], functional["C1rt"],
        functional["C0J2"], functional["C1J2"], functional["C0rdr"],
        functional["C1rdr"], functional["C0nJ"], functional["C1nJ"],
        functional["sigma"]});
  }
}

void EDF::setFromInteraction(SkyrmeParameters skyrme) {
  double t0 = skyrme.t0;
  double t3 = skyrme.t3;
  double x0 = skyrme.x0;
  double x1 = skyrme.x1;
  double x2 = skyrme.x2;
  double x3 = skyrme.x3;
  double t1 = skyrme.t1;
  double t2 = skyrme.t2;
  double W0 = skyrme.W0;
  double W0_pr = skyrme.W0_pr;
  double te = skyrme.te;
  double to = skyrme.to;

  params.sigma = skyrme.sigma;
  params.C0rr = 3 * t0 / 8;
  params.C1rr = -t0 * (0.5 + x0) / 4;
  params.C0Drr = 3 * t3 / 48;
  params.C1Drr = -t3 * (0.5 + x3) / 24;

  params.C0rt = 3 * t1 / 16 + t2 * (5.0 / 4.0 + x2) / 4;
  params.C1rt = -t1 * (0.5 + x1) / 8 + t2 * (0.5 + x2) / 8;

  params.C0rdr = -9 * t1 / 64 + t2 * (5.0 / 4.0 + x2) / 16;
  params.C1rdr = 3 * t1 * (0.5 + x1) / 32 + t2 * (0.5 + x2) / 32;

  params.C0nJ = -0.5 * W0 - 0.25 * W0_pr;
  params.C1nJ = -0.5 * W0;

  double C0T = -(t1 * (1 - 2 * x1) - t2 * (1 + 2 * x2)) / 16;
  double C1T = -(t1 - t2) / 16;

  double C0F = -3 * (te + 3 * to) / 8;
  double C1F = -3 * (te - to) / 8;

  // CtJ2 = -0.5*(CtT - 0.5CtF)
  params.C0J2 = -0.5 * (C0T - 0.5 * C0F);
  params.C1J2 = -0.5 * (C1T - 0.5 * C1F);

  // CtJt = -CtT - 0.5CtF

  // CtJs = -CtT/3 - 2CtF/3
}
