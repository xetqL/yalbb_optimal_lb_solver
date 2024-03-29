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

    template<unsigned N>
    struct Uniform{
        Real temp;
        std::uniform_real_distribution<Real> uniform;
        explicit Uniform(Real temp) : temp(temp), uniform(-2.0*temp*temp, 2.0*temp*temp) {  }

        std::array<Real, N> operator()(std::mt19937 &gen, const std::array<Real, N>& pos) {
            std::array<Real, N> v {  };
            std::generate(v.begin(), v.end(), [&](){return uniform(gen);});
            return v;
        }
    };

    template<int N>
    struct ExpandFromPoint {
        std::array<Real, N> expand_from;
        Real temp;
        std::uniform_real_distribution<Real> uniform;

        explicit ExpandFromPoint(Real temp, std::array<Real, N> expand_from) :
                temp(temp), uniform(0.0, 2.0*temp*temp), expand_from((expand_from)) {
        }

        std::array<Real, N> operator()(std::mt19937 &gen, const std::array<Real, N>& pos) {
            using namespace vec::generic;
            const auto vec = pos - expand_from;
            auto v = normalize(vec) * temp;
            return v;
        }

    };
template<class Real> std::array<std::array<Real, 2>, 2> get_rotation_matrix(Real theta) {
    std::array<std::array<Real, 2>, 2> rotate{};
    rotate[0][0] =  std::cos(theta);
    rotate[0][1] = -std::sin(theta);
    rotate[1][0] =  std::sin(theta);
    rotate[1][1] =  std::cos(theta);
    return rotate;
}
template<class Real>
std::pair<Real, Real> rotate(const std::array<std::array<Real, 2>, 2> &matrix, Real x, Real y) {
    return {matrix[0][0] * x + matrix[0][1] * y, matrix[1][0] * x + matrix[1][1] * y};
}
struct PerpendicularTo {
    std::array<Real, 2> center;
    Real temp;
    std::uniform_real_distribution<Real> uniform;

    explicit PerpendicularTo(Real temp, std::array<Real, 2> rotation_point) :
            temp(temp), uniform(0.0, 2.0*temp*temp), center((rotation_point)) {
    }

    std::array<Real, 2> operator()(std::mt19937 &gen, const std::array<Real, 2>& pos) {
        using namespace vec::generic;
        const auto vec = pos - center;
        auto v = normalize(vec) * temp;
        auto rot = get_rotation_matrix(M_PI / 2.0);
        auto [rx, ry]= rotate(rot, v[0], v[1]);
        v[0] = rx;
        v[1] = ry;
        return v;
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
            using namespace vec::generic;
            Real strength = temp;//uniform(gen);
            std::array<Real, N> vec = expand_from - pos;
            return normalize(vec) * strength;
        }
        std::array<Real, N> operator()(const std::array<Real, N>& pos) {
            using namespace vec::generic;
            std::array<Real, N> vec = expand_from - pos;
            return normalize(vec) * temp;
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
    template<int N> using  UniformOnSphere         = statistic::UniformOnSphereEdgeDistribution<N, Real>;
    template<int N> struct EquidistantOnDisk {
        mutable unsigned i=0;
        Real k;
        std::array<Real, N> center {};
        Real angle;
        explicit EquidistantOnDisk(unsigned start_from, Real k, std::array<Real, N> center, Real angle = 0) : i(start_from), k(k), center(center), angle(angle) {}
        std::array<Real, N> operator()(std::mt19937 &gen) {
            if constexpr(N==3){
                throw std::logic_error("not defined");
            } else {
                const Real gr=(std::sqrt(5.0) + 1.0) / 2.0;  // golden ratio = 1.6180339887498948482
                const Real ga=(2.0 - gr) * (2.0*M_PI);  // golden angle = 2.39996322972865332

                const Real r = sqrt(i) * k;
                const Real theta = ga * i;

                const Real x = std::cos(theta) * r;
                const Real y = std::sin(theta) * r;

                i++;
                Real _theta = angle * M_PI / 180.0 ;

                Real cs = std::cos(_theta);
                Real sn = std::sin(_theta);

                Real px = x * cs - y * sn;
                Real py = x * sn + y * cs;

                return {px + center[0], py + center[1]};
            }
        }
    };
    template<int N> struct Ordered {
        mutable unsigned i=0;
        ;
        std::array<Real, N> center {};
        Real angle;
        Real radius;
        Real el_radius;
        std::array<Real, N> p{};
        explicit Ordered(Real r, std::array<Real, N> center, Real el_radius, Real angle = 0) :
            radius(r), center(center), el_radius(el_radius), angle(angle) {
            using namespace vec::generic;
            p = center - r;
        }

        std::array<Real, N> operator()(std::mt19937 &gen) {
            using namespace vec::generic;
            auto r2 = radius*radius;
            do {
                p[0] += el_radius;

                if(std::abs(p[0]-center[0]) > radius) {
                    p[0] = center[0] - radius;
                    p[1] += el_radius;
                }

                if(std::abs(p[1]-center[1]) > radius) {
                    p[1] = center[1] - radius;
                    if constexpr(N==3)
                        p[2] += el_radius;
                    else {
                        break;
                    }
                }
                if constexpr(N==3) if(std::abs(p[2]-center[2]) > radius) return {-1, -1, -1};
            } while(norm2(p - center) > r2);
            i++;
            return p;
        }
    };
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

template<int N, class  PositionFunctor, class VelocityFunctor>
void generate_random_particles(MESH_DATA<elements::Element<N>>* mesh,
                               int rank, int seed, int npart, PositionFunctor rand_pos, VelocityFunctor rand_vel) {
    if (!rank) {
        mesh->els.reserve(npart);
        std::cout << "Generating data ..." << std::endl;
        std::mt19937 my_gen(seed);
        for(int i = 0;i < npart; ++i) {
            auto pos = rand_pos(my_gen);
            mesh->els.emplace_back(pos, rand_vel(my_gen, pos), i, i);
        }
        std::cout << mesh->els.size() << " Done !" << std::endl;
    }
}

template<int N, class  PositionFunctor, class VelocityFunctor>
void generate_random_particles(MESH_DATA<elements::Element<N>>* mesh,
                           int rank, int seed, int npart, PositionFunctor rand_pos, VelocityFunctor rand_vel, MPI_Comm comm) {
    int wsize;
    MPI_Comm_size(comm, &wsize);
    double per_proc = static_cast<double>(npart) / wsize;
    unsigned part_to_gen;
    if(rank == wsize - 1) {
        part_to_gen = npart - (wsize-1) * static_cast<unsigned>(per_proc);
    } else {
        part_to_gen = per_proc;
    }
    mesh->els.reserve(mesh->els.size() + part_to_gen);
    std::mt19937 my_gen_vel(seed + rank);
    std::mt19937 my_gen_pos(seed + rank + wsize);
    for(int i = 0;i < part_to_gen; ++i) {
        auto pos = rand_pos(my_gen_pos);
        mesh->els.emplace_back(pos, rand_vel(my_gen_vel, pos), i, i);
    }
}

#endif //NBMPI_INITIAL_CONDITIONS_HPP
