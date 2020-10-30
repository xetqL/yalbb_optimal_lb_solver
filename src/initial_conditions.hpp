//
// Created by xetql on 08.06.18.
//

#ifndef NBMPI_INITIAL_CONDITIONS_HPP
#define NBMPI_INITIAL_CONDITIONS_HPP

#include <yalbb/utils.hpp>
#include "spatial_elements.hpp"

#include <memory>
#include <random>

namespace initial_condition {

static std::random_device __rd;
static std::mt19937 __gen(__rd()); //Standard mersenne_twister_engine seeded with rd()

template<class Candidate>
class RejectionCondition {
public:
    virtual bool predicate(const Candidate& c) const = 0;
};

namespace lj {

template<int N>
class RejectionCondition : public initial_condition::RejectionCondition<elements::Element<N>> {
    const std::vector<elements::Element<N>>* others;
public:
    const Real sig;
    const Real min_r2;
    const Real T0;
    const Real xmin;
    const Real ymin;
    const Real zmin;
    const Real xmax;
    const Real ymax;
    const Real zmax;
    const sim_param_t* params;

    RejectionCondition(const std::vector<elements::Element<N>>* others,
                       const Real sig,
                       const Real min_r2,
                       const Real T0,
                       const Real xmin,
                       const Real ymin,
                       const Real zmin,
                       const Real xmax,
                       const Real ymax,
                       const Real zmax,
                       const sim_param_t* params) :
            others(others),
            sig(sig), min_r2(min_r2), T0(T0),
            xmin(xmin), ymin(ymin), zmin(zmin),
            xmax(xmax), ymax(ymax), zmax(zmax), params(params) {}

    bool predicate(const elements::Element<N>& c) const override {
        if constexpr (N > 2) {
            return std::all_of(others->cbegin(), others->cend(), [&](auto o) {
                return xmin < c.position.at(0) && c.position.at(0) < xmax &&
                       ymin < c.position.at(1) && c.position.at(1) < ymax &&
                       zmin < c.position.at(2) && c.position.at(2) < zmax &&
                       elements::distance2<N>(c, o) >= min_r2;
            });
        }else
            return std::all_of(others->cbegin(), others->cend(), [&](auto o) {
                return xmin < c.position.at(0) && c.position.at(0) < xmax &&
                       ymin < c.position.at(1) && c.position.at(1) < ymax &&
                       elements::distance2<N>(c, o) >= min_r2;
            });
    }
};

} // end of namespace lennard_jones

template<int N>
class RandomElementsGenerator {
public:
    virtual void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                                   const std::shared_ptr<lj::RejectionCondition<N>> cond) = 0;
};

namespace lj {


template<int N>
class RandomElementsGen {
    std::shared_ptr<lj::RejectionCondition<N>> cond;
    int seed, max_trial;
public:
    RandomElementsGen(int seed, int max_trial, std::shared_ptr<lj::RejectionCondition<N>> cond) :
        seed(seed), max_trial(max_trial), cond(cond) {}

    template<class PositionDist, class VelocityDist>
    void generate_elements(std::vector<elements::Element<N>>& elements, const int n, PositionDist pdist, VelocityDist vdist)   {
        int number_of_element_generated = 0;
        const Real dblT0Sqr = 2.0 * cond->T0 * cond->T0;
        //std::normal_distribution<Real> temp_dist(0.0, dblT0Sqr);
        //std::uniform_real_distribution<Real> utemp_dist(0.0, dblT0Sqr);
        //Real cx = cond->params->simsize / 2.0,
        //     cy = cond->params->simsize / 2.0,
        //     cz = cond->params->simsize / 2.0;
        //statistic::UniformSphericalDistribution<N, Real> sphere(cond->params->simsize / 3.0, cx, cy, cz);

        std::mt19937 my_gen(seed);

        int trial = 0;
        std::array<Real, N>  element_position, velocity;

        Integer lcxyz;
        std::array<Integer, N> lc;
        Real cut_off = cond->params->rc;
        lc[0] = (cond->xmax - cond->xmin) / cut_off;
        lc[1] = (cond->ymax - cond->ymin) / cut_off;
        lcxyz = lc[0] * lc[1];
        if constexpr (N==3){
            lc[2] = (cond->zmax - cond->zmin) / cut_off;
            lcxyz *= lc[2];
        }
        const Integer EMPTY = -1;
        std::vector<Integer> head(lcxyz, -1), lscl(n, -1);
        Integer generated = elements.size();


        while(generated < n) {
            while(trial < max_trial) {

                element_position = pdist(my_gen);
                velocity         = vdist(my_gen);

                auto element = elements::Element<N>(element_position, velocity, generated, generated);

                std::array<Real, 3> delta_dim;
                Integer c, c1,  j;
                std::array<Integer, N> ic, ic1;
                elements::Element<N> receiver;

                c = position_to_cell<N>(element.position, cut_off, lc[0], lc[1]);

                for(int d = 0; d < N; ++d)
                    ic[d] = (element.position[d]) / cut_off;

                bool accepted = true;
                for (ic1[0] = (ic[0] - 1); ic1[0] <= (ic[0]+1); ic1[0]++) {
                    for (ic1[1] = (ic[1] - 1); ic1[1] <= (ic[1] + 1); ic1[1]++) {
                        if constexpr(N==3) {
                            for (ic1[2] = (ic[2] - 1); ic1[2] <= (ic[2] + 1); ic1[2]++) {
                                if ((ic1[0] < 0 || ic1[0] >= lc[0])
                                    ||  (ic1[1] < 0 || ic1[1] >= lc[1]) ||
                                    (ic1[2] < 0 || ic1[2] >= lc[2])) {
                                    continue;
                                }
                                c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);

                                j = head[c1];

                                while (j != EMPTY && accepted) {
                                    receiver = elements[j];
                                    if(elements::distance2(receiver, element) <= cond->min_r2) {
                                        accepted = false;
                                    }
                                    j = lscl[j];
                                }
                            }
                        }else{
                            if ((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1])) {
                                continue;
                            }
                            c1 = (ic1[0]) + (lc[0] * ic1[1]);

                            j = head[c1];

                            while (j != EMPTY && accepted) {
                                receiver = elements[j];
                                if(elements::distance2<N>(receiver, element) <= cond->min_r2) {
                                    accepted = false;
                                }
                                j = lscl[j];
                            }
                        }
                    }
                }
                if(accepted){
                    trial = 0;
                    CLL_append<N>(generated, c, element, &head, &lscl);
                    elements.push_back(element);
                    generated = elements.size();
                    break;
                } else {
                    trial++;
                }

            }
            if(trial == max_trial)
                return; // when you cant generate new particles with less than max trials stop.
        }
    }
};

template<int N>
class UniformRandomElementsGenerator : public RandomElementsGenerator<N> {
    int seed;

    const int max_trial;
public:
    UniformRandomElementsGenerator(int seed, const int max_trial = 10000) : seed(seed), max_trial(max_trial) {}

    void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                           const std::shared_ptr<lj::RejectionCondition<N>> condition) override {
        int number_of_element_generated = 0;
        const Real dblT0Sqr = 2.0 * condition->T0 * condition->T0;
        std::normal_distribution<Real> temp_dist(0.0, dblT0Sqr);
        std::uniform_real_distribution<Real> utemp_dist(0.0, dblT0Sqr);

        std::uniform_real_distribution<Real>
            udistx(condition->xmin, condition->xmax),
            udisty(condition->ymin, condition->ymax),
            udistz(condition->zmin, condition->zmax);

        std::mt19937 my_gen(seed);
        int trial = 0;
        std::array<Real, N>  element_position, velocity;

        Integer lcxyz;
        std::array<Integer, N> lc;
        Real cut_off = condition->params->rc;
        lc[0] = (condition->xmax - condition->xmin) / cut_off;
        lc[1] = (condition->ymax - condition->ymin) / cut_off;
        lcxyz = lc[0] * lc[1];
        if constexpr (N==3){
            lc[2] = (condition->zmax - condition->zmin) / cut_off;
            lcxyz *= lc[2];
        }
        const Integer EMPTY = -1;
        std::vector<Integer> head(lcxyz, -1), lscl(n, -1);
        Integer generated = elements.size();
        std::array<Real, N> singularity;
        std::generate(singularity.begin(), singularity.end(), [&my_gen, &udist=udistx](){return udist(my_gen);});
        while(generated < n) {
            while(trial < max_trial) {
                if constexpr (N==3) {
                    element_position = { udistx(my_gen), udisty(my_gen), udistz(my_gen) };
                    auto strength    = utemp_dist(my_gen);
                    velocity         = {
                            ((condition->xmin + singularity[0]) - element_position[0]) * strength,
                            ((condition->ymin + singularity[1]) - element_position[1]) * strength,
                            ((condition->zmin + singularity[2]) - element_position[2]) * strength
                    };
                } else {
                    auto strength    = utemp_dist(my_gen);
                    element_position = { udistx(my_gen), udisty(my_gen)};
                    velocity         = {
                            ((condition->xmin + singularity[0]) - element_position[0]) * strength,
                            ((condition->ymin + singularity[1]) - element_position[1]) * strength,
                    };
                }

                auto element = elements::Element<N>(element_position, velocity, generated, generated);

                std::array<Real, 3> delta_dim;
                Integer c, c1,  j;
                std::array<Integer, N> ic, ic1;
                elements::Element<N> receiver;

                c = position_to_cell<N>(element.position, cut_off, lc[0], lc[1]);

                for(int d = 0; d < N; ++d)
                    ic[d] = (element.position[d]) / cut_off;

                bool accepted = true;
                for (ic1[0] = (ic[0] - 1); ic1[0] <= (ic[0]+1); ic1[0]++) {
                    for (ic1[1] = (ic[1] - 1); ic1[1] <= (ic[1] + 1); ic1[1]++) {
                        if constexpr(N==3) {
                            for (ic1[2] = (ic[2] - 1); ic1[2] <= (ic[2] + 1); ic1[2]++) {
                                if ((ic1[0] < 0 || ic1[0] >= lc[0])
                                ||  (ic1[1] < 0 || ic1[1] >= lc[1]) ||
                                    (ic1[2] < 0 || ic1[2] >= lc[2])) {
                                    continue;
                                }
                                c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);

                                j = head[c1];

                                while (j != EMPTY && accepted) {
                                   receiver = elements[j];
                                   if(elements::distance2(receiver, element) <= condition->min_r2) {
                                       accepted = false;
                                   }
                                   j = lscl[j];
                                }
                            }
                        }else{
                            if ((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1])) {
                                continue;
                            }
                            c1 = (ic1[0]) + (lc[0] * ic1[1]);

                            j = head[c1];

                            while (j != EMPTY && accepted) {
                                receiver = elements[j];
                                if(elements::distance2<N>(receiver, element) <= condition->min_r2) {
                                    accepted = false;
                                }
                                j = lscl[j];
                            }
                        }
                    }
                }
                if(accepted){
                    trial = 0;
                    CLL_append<N>(generated, c, element, &head, &lscl);
                    elements.push_back(element);
                    generated = elements.size();
                    break;
                } else {
                    trial++;
                }

            }
            if(trial == max_trial)
                return; // when you cant generate new particles with less than max trials stop.
        }
    }
};

template<int N>
class SphericalRandomElementsGenerator : public RandomElementsGenerator<N> {
    int seed;

    const int max_trial;
public:
    SphericalRandomElementsGenerator(int seed, const int max_trial = 10000) : seed(seed), max_trial(max_trial) {}

    void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                           const std::shared_ptr<lj::RejectionCondition<N>> condition) override {
        int number_of_element_generated = 0;
        const Real dblT0Sqr = 2.0 * condition->T0 * condition->T0;
        std::normal_distribution<Real> temp_dist(0.0, dblT0Sqr);
        std::uniform_real_distribution<Real> utemp_dist(0.0, dblT0Sqr);
        Real cx = condition->params->simsize / 2.0, cy = condition->params->simsize / 2.0, cz = condition->params->simsize / 2.0;
        statistic::UniformSphericalDistribution<N, Real> sphere(condition->params->simsize / 3.0, cx, cy, cz);

        std::mt19937 my_gen(seed);
        int trial = 0;
        std::array<Real, N>  element_position, velocity;

        Integer lcxyz;
        std::array<Integer, N> lc;
        Real cut_off = condition->params->rc;
        lc[0] = (condition->xmax - condition->xmin) / cut_off;
        lc[1] = (condition->ymax - condition->ymin) / cut_off;
        lcxyz = lc[0] * lc[1];
        if constexpr (N==3){
            lc[2] = (condition->zmax - condition->zmin) / cut_off;
            lcxyz *= lc[2];
        }
        const Integer EMPTY = -1;
        std::vector<Integer> head(lcxyz, -1), lscl(n, -1);
        Integer generated = elements.size();
        std::array<Real, N> singularity = {cx, cy, cz};

        while(generated < n) {
            while(trial < max_trial) {
                if constexpr (N==3) {
                    element_position = sphere(my_gen);
                    auto strength    = utemp_dist(my_gen);
                    velocity         = {
                            ((condition->xmin + singularity[0]) - element_position[0]) * strength,
                            ((condition->ymin + singularity[1]) - element_position[1]) * strength,
                            ((condition->zmin + singularity[2]) - element_position[2]) * strength
                    };
                } else {
                    auto strength    = utemp_dist(my_gen);
                    element_position = sphere(my_gen);
                    velocity         = {
                            ((condition->xmin + singularity[0]) - element_position[0]) * strength,
                            ((condition->ymin + singularity[1]) - element_position[1]) * strength,
                    };
                }

                auto element = elements::Element<N>(element_position, velocity, generated, generated);

                std::array<Real, 3> delta_dim;
                Integer c, c1,  j;
                std::array<Integer, N> ic, ic1;
                elements::Element<N> receiver;

                c = position_to_cell<N>(element.position, cut_off, lc[0], lc[1]);

                for(int d = 0; d < N; ++d)
                    ic[d] = (element.position[d]) / cut_off;

                bool accepted = true;
                for (ic1[0] = (ic[0] - 1); ic1[0] <= (ic[0]+1); ic1[0]++) {
                    for (ic1[1] = (ic[1] - 1); ic1[1] <= (ic[1] + 1); ic1[1]++) {
                        if constexpr(N==3) {
                            for (ic1[2] = (ic[2] - 1); ic1[2] <= (ic[2] + 1); ic1[2]++) {
                                if ((ic1[0] < 0 || ic1[0] >= lc[0])
                                    ||  (ic1[1] < 0 || ic1[1] >= lc[1]) ||
                                    (ic1[2] < 0 || ic1[2] >= lc[2])) {
                                    continue;
                                }
                                c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);

                                j = head[c1];

                                while (j != EMPTY && accepted) {
                                    receiver = elements[j];
                                    if(elements::distance2(receiver, element) <= condition->min_r2) {
                                        accepted = false;
                                    }
                                    j = lscl[j];
                                }
                            }
                        }else{
                            if ((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1])) {
                                continue;
                            }
                            c1 = (ic1[0]) + (lc[0] * ic1[1]);

                            j = head[c1];

                            while (j != EMPTY && accepted) {
                                receiver = elements[j];
                                if(elements::distance2<N>(receiver, element) <= condition->min_r2) {
                                    accepted = false;
                                }
                                j = lscl[j];
                            }
                        }
                    }
                }
                if(accepted){
                    trial = 0;
                    CLL_append<N>(generated, c, element, &head, &lscl);
                    elements.push_back(element);
                    generated = elements.size();
                    break;
                } else {
                    trial++;
                }

            }
            if(trial == max_trial)
                return; // when you cant generate new particles with less than max trials stop.
        }
    }
};

template<int N>
class ParticleWallRandomElementsGenerator : public RandomElementsGenerator<N> {
        int seed;

        const int max_trial;
    public:
    ParticleWallRandomElementsGenerator(int seed, const int max_trial = 10000) : seed(seed), max_trial(max_trial) {}

        void generate_elements(std::vector<elements::Element<N>>& elements, const int n,
                               const std::shared_ptr<lj::RejectionCondition<N>> condition) override {
            int number_of_element_generated = 0;

            const Real dblT0Sqr = 2.0 * condition->T0 * condition->T0;

            std::uniform_real_distribution<Real> utemp_dist(0.0, dblT0Sqr);

            std::uniform_real_distribution<Real>
                    udistx(condition->xmin, condition->xmax),
                    udisty(condition->ymin, condition->ymax),
                    udistz(condition->zmax/2.0-3.2*condition->params->rc, condition->zmax/2.0+3.2*condition->params->rc);

            std::mt19937 my_gen(seed);
            int trial = 0;
            std::array<Real, N>  element_position, velocity;

            Integer lcxyz;
            std::array<Integer, N> lc;
            Real cut_off = condition->params->rc;
            lc[0] = (condition->xmax - condition->xmin) / cut_off;
            lc[1] = (condition->ymax - condition->ymin) / cut_off;
            lcxyz = lc[0] * lc[1];
            if constexpr (N==3){
                lc[2] = (condition->zmax - condition->zmin) / cut_off;
                lcxyz *= lc[2];
            }
            const Integer EMPTY = -1;
            std::vector<Integer> head(lcxyz, -1), lscl(n, -1);
            Integer generated = elements.size();
            std::array<Real, N> singularity;
            std::generate(singularity.begin(), singularity.end(), [&my_gen, &udist=udistx](){return udist(my_gen);});
            while(generated < n) {
                while(trial < max_trial) {
                    if constexpr (N==3) {
                        element_position = { udistx(my_gen), udisty(my_gen), udistz(my_gen) };
                        auto strength    = utemp_dist(my_gen);
                        velocity         = {
                                ((condition->xmin + singularity[0]) - element_position[0]) * strength,
                                ((condition->ymin + singularity[1]) - element_position[1]) * strength,
                                ((condition->zmin + singularity[2]) - element_position[2]) * strength
                        };
                    } else {
                        auto strength    = utemp_dist(my_gen);
                        element_position = { udistx(my_gen), udisty(my_gen)};
                        velocity         = {
                                ((condition->xmin + singularity[0]) - element_position[0]) * strength,
                                ((condition->ymin + singularity[1]) - element_position[1]) * strength,
                        };
                    }

                    auto element = elements::Element<N>(element_position, velocity, generated, generated);

                    std::array<Real, 3> delta_dim;
                    Integer c, c1,  j;
                    std::array<Integer, N> ic, ic1;
                    elements::Element<N> receiver;

                    c = position_to_cell<N>(element.position, cut_off, lc[0], lc[1]);

                    for(int d = 0; d < N; ++d)
                        ic[d] = (element.position[d]) / cut_off;

                    bool accepted = true;
                    for (ic1[0] = (ic[0] - 1); ic1[0] <= (ic[0]+1); ic1[0]++) {
                        for (ic1[1] = (ic[1] - 1); ic1[1] <= (ic[1] + 1); ic1[1]++) {
                            if constexpr(N==3) {
                                for (ic1[2] = (ic[2] - 1); ic1[2] <= (ic[2] + 1); ic1[2]++) {
                                    if ((ic1[0] < 0 || ic1[0] >= lc[0])
                                        ||  (ic1[1] < 0 || ic1[1] >= lc[1]) ||
                                        (ic1[2] < 0 || ic1[2] >= lc[2])) {
                                        continue;
                                    }
                                    c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);

                                    j = head[c1];

                                    while (j != EMPTY && accepted) {
                                        receiver = elements[j];
                                        if(elements::distance2(receiver, element) <= condition->min_r2) {
                                            accepted = false;
                                        }
                                        j = lscl[j];
                                    }
                                }
                            }else{
                                if ((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1])) {
                                    continue;
                                }
                                c1 = (ic1[0]) + (lc[0] * ic1[1]);

                                j = head[c1];

                                while (j != EMPTY && accepted) {
                                    receiver = elements[j];
                                    if(elements::distance2<N>(receiver, element) <= condition->min_r2) {
                                        accepted = false;
                                    }
                                    j = lscl[j];
                                }
                            }
                        }
                    }
                    if(accepted){
                        trial = 0;
                        CLL_append<N>(generated, c, element, &head, &lscl);
                        elements.push_back(element);
                        generated = elements.size();
                        break;
                    } else {
                        trial++;
                    }

                }
                if(trial == max_trial)
                    return; // when you cant generate new particles with less than max trials stop.
            }
        }
    };

} // end of namespace lennard_jones


} // end of namespace initial_condition
namespace vel {
    template<int N>
    struct ExpandFromPoint {
        std::array<Real, N> expand_from;
        Real temp;
        std::uniform_real_distribution<Real> uniform;

        explicit ExpandFromPoint(Real temp, std::array<Real, N> expand_from) :
                temp(temp), uniform(0.0, 2.0*temp*temp), expand_from((expand_from)) {
        }

        std::array<Real, N> operator()(std::mt19937 &gen, const std::array<Real, N>& pos) {
            auto strength = uniform(gen);
            std::array<Real, N> vec;
            for(int i = 0; i < N; ++i) vec[i] = pos[i] - expand_from[i];
            Real length = std::sqrt(std::accumulate(vec.begin(), vec.end(), (Real) 0.0, [](auto p, auto v){return p + v*v;}));
            if constexpr (N==3)
                return {
                        ((vec[0] / length)) * strength,
                        ((vec[1] / length)) * strength,
                        ((vec[2] / length)) * strength
                };
            else
                return { 0.0f, 0.0f };
        }
    };
    template<int N>
    struct ContractToPoint {
        std::array<Real, N> expand_from;
        Real temp;
        std::uniform_real_distribution<Real> uniform;
        explicit ContractToPoint(Real temp, std::array<Real, N> expand_from) :
                temp(temp), uniform(0.0, 2.0*temp*temp), expand_from(std::move(expand_from)) {}

        std::array<Real, N> operator()(std::mt19937 &gen, const std::array<Real, N>& pos) {
            Real strength = uniform(gen);
            std::array<Real, N> vec;
            for(int i = 0; i < N; ++i) {
                vec[i] = expand_from[i] - pos[i];
            }
            Real length = std::sqrt(std::accumulate(vec.begin(), vec.end(), (Real) 0.0, [](auto p, auto v){return p + v*v;}));
            if constexpr (N==3)
                return {
                        ((vec[0] / length)) * strength,
                        ((vec[1] / length)) * strength,
                        ((vec[2] / length)) * strength
                };
            else
                return { 0.0f, 0.0f };
        }
    };

    template<int N, unsigned Axis>
    struct ParallelToAxis {
        static_assert(Axis < N);
        const Real p;
        ParallelToAxis(Real p) : p(p) {}
        constexpr std::array<Real, N> operator()(std::mt19937 &gen, const std::array<Real, N>& pos) {
            std::array<Real, N> vel {};
            vel.at(Axis) = p;
            return vel;
        }
    };

    template<int N, unsigned Axis>
    struct GoToStripe {
        static_assert(Axis < N);
        const Real s;
        const Real top;
        constexpr std::array<Real, N> operator()(std::mt19937 &gen, const std::array<Real, N>& pos) {
            std::array<Real, N> vel {};
            vel.at(Axis) = s * std::max((Real) 0.0, top - pos.at(Axis));
            return vel;
        }
    };

    template<int N>
    struct None {
        constexpr std::array<Real, N> operator()(std::mt19937 &gen, const std::array<Real, N>& pos) {
            std::array<Real, N> vel {};
            return vel;
        }
    };

}
namespace pos {
    template<int N> struct UniformInCube {
        std::array<Real, 2*N> dimension;
        std::uniform_real_distribution<Real> uniform{(Real) 0.0, (Real) 1.0};
        explicit UniformInCube(std::array<Real, 2*N> dimension) : dimension(std::move(dimension)) {}
        std::array<Real, N> operator()(std::mt19937 &gen) {
            if constexpr(N==3){
                return { uniform(gen) * (dimension.at(1)-dimension.at(0)) + dimension.at(0),
                         uniform(gen) * (dimension.at(3)-dimension.at(2)) + dimension.at(2),
                         uniform(gen) * (dimension.at(5)-dimension.at(4)) + dimension.at(4)};
            } else {
                return { uniform(gen) * (dimension.at(1)-dimension.at(0)) + dimension.at(0),
                         uniform(gen) * (dimension.at(3)-dimension.at(2)) + dimension.at(2)};
            }
        }
    };
    template<int N> using  UniformInSphere = statistic::UniformSphericalDistribution<N, Real>;
    template<int N> using  UniformOnSphere = statistic::UniformOnSphereEdgeDistribution<N, Real>;
    template<int N> struct LoadFromFile {
        std::ifstream infile;
        std::string   header;
        LoadFromFile(std::string fname) : infile(fname) {
                std::getline(infile, header);
        }
        std::array<Real, N> operator()(std::mt19937 &gen) {
            return io::load_one<N>(infile);
        }
    };
    template<int N> struct Fixed {
        Real p;
        std::array<Real, N> operator()(std::mt19937 &gen) {
            if constexpr(N==3) return {p, p, p}; else return {p, p};
        }
    };
}

template<int N, class  PositionFunctor, class VelocityFunctor>
MESH_DATA<elements::Element<N>> generate_random_particles(int rank, const sim_param_t& params, PositionFunctor rand_pos, VelocityFunctor rand_vel) {
    MESH_DATA<elements::Element<N>> mesh;
    if (!rank) {
        mesh.els.reserve(params.npart);
        std::cout << "Generating data ..." << std::endl;
        std::mt19937 my_gen(params.seed);
        for(int i = 0;i < params.npart; ++i) {
            auto pos = rand_pos(my_gen);
            mesh.els.emplace_back(pos, rand_vel(my_gen, pos), i, i);
        }
        std::cout << mesh.els.size() << " Done !" << std::endl;
    }
    return mesh;
}

#endif //NBMPI_INITIAL_CONDITIONS_HPP
