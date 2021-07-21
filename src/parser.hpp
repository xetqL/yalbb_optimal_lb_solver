//
// Created by xetql on 4/15/21.
//

#pragma once

#include <yalbb/params.hpp>

struct param_t : public sim_param_t {
    float eps_lj{};       /* Strength for L-J (1)       */
    float sig_lj{};       /* Radius for L-J   (1e-2)    */
    float G{};            /* Gravitational strength (1) */
    float T0{};           /* Initial temperature (1)    */
    float bounce{};       /* shock absorption factor (0=no bounce, 1=full bounce) */
};

template<class T>
void print_params(T& stream, const param_t* params) {
    stream << "[Global]" << std::endl;
    stream << show(params->simulation_name) << std::endl;
    stream << show(params->npart) << std::endl;
    stream << show(params->seed) << std::endl;
    stream << show(params->id) << std::endl;
    stream << "\n";

    stream << "[Physics]" << std::endl;
    stream << show(params->T0) << std::endl;
    stream << show(params->sig_lj) << std::endl;
    stream << show(params->eps_lj) << std::endl;
    stream << show(params->G) << std::endl;
    stream << show(params->bounce) << std::endl;
    stream << show(params->dt) << std::endl;
    stream << show(params->rc) << std::endl;
    stream << "\n";

    stream << "[Box]" << std::endl;
    stream << show(params->simsize) << std::endl;
    stream << "\n";

    stream << "[Iterations]" << std::endl;
    stream << show(params->nframes) << std::endl;
    stream << show(params->npframe) << std::endl;
    stream << "Total=" << (params->nframes * params->npframe) << std::endl;
    stream << "\n";

    stream << "[LBSolver]" << std::endl;
    stream << show(params->nb_best_path) << std::endl;
    stream << "\n";

    stream << "[Storing]" << std::endl;
    stream << show(params->monitor) << std::endl;
    stream << show(params->record) << std::endl;
    stream << "\n";

    stream << "[Miscellaneous]" << std::endl;
    stream << show(params->verbosity) << std::endl;
}
struct Parser : public TParser<param_t> {
    param_t* custom_params;
    Parser() {
        custom_params = (param_t*) params.get();
        // Force (user-defined)
        parser.add_opt_value('e', "epslj",       custom_params->eps_lj, 1.0f, "Epsilon (lennard-jones)", "FLOAT");
        parser.add_opt_value('g', "gravitation", custom_params->G, 1.0f, "Number of G's", "FLOAT");
        parser.add_opt_value('T', "temperature", custom_params->T0, 1.0f, "Initial temperatore", "FLOAT");
        parser.add_opt_value('s', "siglj",       custom_params->sig_lj, 1e-2f, "Sigma (lennard-jones)", "FLOAT");
        parser.add_opt_value('b', "bounce",      custom_params->bounce, 1.0f, "Bouncing factor (1=full bounce, 0=no bounce)", "FLOAT");
    }
    void post_parsing() override {
         custom_params->G  *= 9.81f;
         custom_params->rc *= custom_params->sig_lj;
    }
};