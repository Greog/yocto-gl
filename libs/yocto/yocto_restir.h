#include <assert.h>

static vec4f fix_radiance(const vec3f& radiance, const trace_params& params) {
  if (!isfinite(radiance)) { return {0.0f, 0.0f, 0.0f, 1.0f}; }
  if (max(radiance) > params.clamp) {
    vec3f tmp = radiance * (params.clamp / max(radiance));
    return {tmp.x, tmp.y, tmp.z, 1.0f};
  }
  return {radiance.x, radiance.y, radiance.z, 1.0f};
}

static shading_point make_shading_point(const bvh_intersection& intersection,
    const vec3f& outgoing, const trace_scene* scene) {
  auto instance  = scene->instances[intersection.instance];
  auto element   = intersection.element;
  auto uv        = intersection.uv;
  auto point     = shading_point{};
  point.position = eval_position(instance, element, uv);
  point.normal   = eval_shading_normal(instance, element, uv, outgoing);
  point.emission = eval_emission(instance, element, uv, point.normal, outgoing);
  point.outgoing = outgoing;
  point.bsdf     = eval_bsdf(instance, element, uv, point.normal, outgoing);
  // auto opacity  = eval_opacity(instance, element, uv, normal, outgoing);
  return point;
}

static bool is_point_visible(const vec3f& position, const vec3f& point,
    const trace_scene* scene, const trace_bvh* bvh, float threshold = 0.001) {
  auto incoming            = normalize(point - position);
  auto shadow_ray          = ray3f{position, incoming};
  auto shadow_intersection = intersect_bvh(bvh, shadow_ray);
  if (!shadow_intersection.hit) {
    // printf("[warning] Shadow ray hitting nothing!\n");
    return false;
  }

  auto instance = scene->instances[shadow_intersection.instance];
  auto element  = shadow_intersection.element;
  auto uv       = shadow_intersection.uv;
  if (length(point - eval_position(instance, element, uv)) < threshold) {
    return true;
  }
  return false;
}

// light_point, pdf
static pair<light_point, float> sample_area_lights(const trace_scene* scene,
    const trace_lights* lights, float rl, float rel, const vec2f& ruv) {
  auto pdf      = 1.0f;
  auto light_id = sample_uniform((int)lights->lights.size(), rl);
  pdf *= sample_uniform_pdf((int)lights->lights.size());

  auto light = lights->lights[light_id];
  if (light->instance == nullptr) {
    assert(0 && "environments not supported for now");
    return {};
  }

  auto instance = light->instance;
  auto element  = sample_discrete_cdf(light->elements_cdf, rel);
  auto uv = (!instance->shape->triangles.empty()) ? sample_triangle(ruv) : ruv;
  auto point     = light_point{};
  point.position = eval_position(light->instance, element, uv);
  point.normal   = eval_normal(light->instance, element, uv);
  point.emission = eval_emission(instance, element, uv, point.normal, {});

  auto area = light->elements_cdf.back();
  pdf /= area;
  return {point, pdf};
}

static float geometric_term(const vec3f& position, const vec3f& lposition,
    const vec3f& lnormal, const vec3f& incoming) {
  return abs(dot(lnormal, -incoming)) / distance_squared(position, lposition);
}

static vec3f shade_point(const shading_point& point,
    const restir_reservoir& reservoir, const vec3f& outgoing,
    const vec3f& incoming) {
  vec3f bsdfcos = eval_bsdfcos(point.bsdf, point.normal, outgoing, incoming);
  float gterm = geometric_term(
      point.position, reservoir.lpoint.position, reservoir.lpoint.normal, incoming);
  return reservoir.weight * reservoir.lpoint.emission * bsdfcos * gterm;
}

static float eval_p_hat_q_novis(
    const shading_point& point, const light_point& lpoint,
    const trace_scene* scene, const trace_bvh* bvh) {
  vec3f incoming = normalize(lpoint.position - point.position);
  vec3f bsdfcos  = eval_bsdfcos(
      point.bsdf, point.normal, point.outgoing, incoming);
  float gterm    = geometric_term(
      point.position, lpoint.position, lpoint.normal, incoming);
  float p_hat_q  = max(bsdfcos * lpoint.emission) * gterm;
  return p_hat_q;
}

static float eval_p_hat_q_vis(
    const shading_point& point, const light_point& lpoint,
    const trace_scene* scene, const trace_bvh* bvh) {
  if (!is_point_visible(point.position, lpoint.position, scene, bvh)) {
    return 0.0f;
  }
  return eval_p_hat_q_novis(point, lpoint, scene, bvh);
}

static restir_reservoir make_reservoir(bool visibility,
    const shading_point& point, const trace_scene* scene,
    const trace_lights* lights, rng_state& rng, int num_candidates,
    const trace_bvh* bvh) {
  restir_reservoir res           = {};
  float            w_sum         = 0.0f;
  float            sampled_p_hat = 0.0f;
  res.point = point;
  auto eval_p_hat_q =
    (visibility) ? &eval_p_hat_q_vis
                 : &eval_p_hat_q_novis;

  for (int i = 0; i < num_candidates; i++) {
    auto [candidate, candidate_pdf] = sample_area_lights(
        scene, lights, rand1f(rng), rand1f(rng), rand2f(rng));
    float p_hat = eval_p_hat_q(point, candidate, scene, bvh);
    float w     = p_hat / candidate_pdf;
    assert(isfinite(w) && "'candidate_pdf' shall be nonzero");

    res.num_candidates += 1;
    if (w <= 0.0f) { continue; }
    w_sum += w;
    if (rand1f(rng) <= (w / w_sum)) {
      res.lpoint    = candidate;
      sampled_p_hat = p_hat;
    }
  }

  if (sampled_p_hat > 0.0f && num_candidates > 0) {
    res.weight = (1.0f / sampled_p_hat) * (1.0f / num_candidates) * w_sum;
  }

  return res;
}

static restir_reservoir combine_reservoirs(
    bool visibility, bool unbiased, const shading_point& point,
    const vector<restir_reservoir*>& reservoirs, rng_state& rng,
    const trace_scene* scene, const trace_bvh* bvh) {
  restir_reservoir        res             = {};
  float                   w_sum           = 0.0f;
  restir_reservoir*       sampled_res     = nullptr;
  float                   sampled_p_hat_q = 0.0f;
  res.point = point;
  auto eval_p_hat_q =
    (visibility) ? &eval_p_hat_q_vis
                 : &eval_p_hat_q_novis;

  for (int i = 0; i < reservoirs.size(); i++) {
    auto r = reservoirs[i];
    res.num_candidates += r->num_candidates;
    if (r->weight <= 0.0f) { continue; }
    float p_hat_q = eval_p_hat_q(point, r->lpoint, scene, bvh);
    float w       = p_hat_q * r->weight * r->num_candidates;
    if (w <= 0.0f) { continue; }

    w_sum += w;
    if (rand1f(rng) <= (w / w_sum)) {
      sampled_res     = r;
      sampled_p_hat_q = p_hat_q;
    }
  }

  if (sampled_res == nullptr || sampled_p_hat_q == 0.0f) { return res; }
  res.lpoint = sampled_res->lpoint;

  if (!unbiased) {
    res.weight = (1.0f / sampled_p_hat_q) * (1.0f / res.num_candidates) * w_sum;
  }
  else {
    float Z = 0.0f;
    for (int i = 0; i < reservoirs.size(); i++) {
      restir_reservoir* r = reservoirs[i];
      float p_hat_q_i = eval_p_hat_q(r->point, res.lpoint, scene, bvh);
      if (p_hat_q_i > 0.0f) {
        Z += r->num_candidates;
      }
    }

    float m       = 1.0f / Z;
    float p_hat_q = eval_p_hat_q(res.point, res.lpoint, scene, bvh);
    if (Z > 0.0f && p_hat_q > 0.0f) {
      res.weight = (1.0f / p_hat_q) * (m * w_sum);
    }
  }

  return res;
}

void pick_spatial_neighbours(trace_state* state, const vec2i& ij_base,
                             std::vector<restir_reservoir*>& reservoirs) {
  int radius       = 30;
  vec2i image_size = state->render.imsize();
  rng_state& rng   = state->rngs[ij_base];

  for (int i = 0; i < 5; i++) {
    vec2f ij_offset = sample_disk(rand2f(rng)) * radius;
    vec2i ij = ij_base + vec2i{(int)ij_offset.x, (int)ij_offset.y};
    if (ij.x < 0 || ij.y < 0 || ij.x >= image_size.x || ij.y >= image_size.y) {
      continue;
    }
    restir_reservoir* r = &state->reservoirs[ij];
    reservoirs.push_back(r);
  }
}

static vec4f trace_direct(const trace_scene* scene, const trace_bvh* bvh,
    const trace_lights* lights, const ray3f& ray_, rng_state& rng,
    const trace_params& params) {
  // initialize
  auto ray      = ray_;
  auto hit      = !params.envhidden && !scene->environments.empty();
  auto radiance = zero3f;

  // intersect next point
  auto intersection = intersect_bvh(bvh, ray);
  if (!intersection.hit) {
    if (!params.envhidden) {
      radiance = eval_environment(scene, ray.d);
    }
    return {radiance.x, radiance.y, radiance.z, 1};
  }
  hit = true;

  // prepare shading point
  auto outgoing = -ray.d;
  auto point    = make_shading_point(intersection, outgoing, scene);

  // accumulate emission
  radiance += eval_emission(point.emission, point.normal, outgoing);

  // handle delta
  if (is_delta(point.bsdf)) return {radiance.x, radiance.y, radiance.z, 1};

  // sample point on light
  auto [lpoint, pdf] = sample_area_lights(
      scene, lights, rand1f(rng), rand1f(rng), rand2f(rng));
  auto incoming = normalize(lpoint.position - point.position);
  auto bsdfcos  = eval_bsdfcos(point.bsdf, point.normal, outgoing, incoming);
  auto weight   = bsdfcos / pdf;

  // check weight
  if (weight == zero3f || !isfinite(weight))
    return {radiance.x, radiance.y, radiance.z, 1};

  // shadow ray
  auto shadow_ray          = ray3f{point.position, incoming};
  auto shadow_intersection = intersect_bvh(bvh, shadow_ray);

  if (!is_point_visible(point.position, lpoint.position, scene, bvh)) {
    return {radiance.x, radiance.y, radiance.z, 1};
  }

  auto light_point = make_shading_point(
      shadow_intersection, -shadow_ray.d, scene);

  // TODO(giacomo): refactor this into a function.
  auto geometric_term = abs(dot(light_point.normal, -shadow_ray.d)) /
                        distance_squared(point.position, light_point.position);
  weight *= geometric_term;
  radiance += weight * light_point.emission;

  return {radiance.x, radiance.y, radiance.z, 1};
}

static vec3f trace_restir(const trace_scene* scene, const trace_bvh* bvh,
    const trace_lights* lights, const ray3f& ray_, const vec2i& ij,
    trace_state* state, const trace_params& params) {
  // initialize
  auto& rng      = state->rngs[ij];
  auto  ray      = ray_;
  auto  hit      = !params.envhidden && !scene->environments.empty();
  auto  radiance = zero3f;

  // intersect next point
  auto intersection = intersect_bvh(bvh, ray);
  if (!intersection.hit) {
    if (!params.envhidden) {
      return eval_environment(scene, ray.d);
    }
    return zero3f;
  }
  hit = true;

  // prepare shading point
  auto outgoing = -ray.d;
  auto point    = make_shading_point(intersection, outgoing, scene);

  // accumulate emission
  radiance += eval_emission(point.emission, point.normal, outgoing);

  // handle delta
  if (is_delta(point.bsdf)) return radiance;

  // sample incoming direction
  restir_reservoir& reservoir = state->reservoirs[ij];

  if (params.restir_type == "noreuse") {
    reservoir = make_reservoir(params.restir_vis, point, scene, lights, rng,
                               params.restir_candidates, bvh);
  }
  else if (params.restir_type == "split") {
    auto r1 = make_reservoir(params.restir_vis, point, scene, lights, rng, 4,
                             bvh);
    auto r2 = make_reservoir(params.restir_vis, point, scene, lights, rng, 8,
                             bvh);
    auto r3 = make_reservoir(params.restir_vis, point, scene, lights, rng, 12,
                            bvh);
    auto r4 = make_reservoir(params.restir_vis, point, scene, lights, rng, 16,
                             bvh);
    auto r5 = make_reservoir(params.restir_vis, point, scene, lights, rng, 20,
                             bvh);
    reservoir = combine_reservoirs(params.restir_vis, params.restir_unbias,
                                   point, {&r1, &r2, &r3, &r4, &r5}, rng,
                                   scene, bvh);
  }
  else if (params.restir_type == "temporal") {
    auto curr_res = make_reservoir(params.restir_vis, point, scene, lights, rng,
                                   params.restir_candidates, bvh);
    auto prev_res = state->reservoirs[ij];
    if (!is_point_visible(point.position, curr_res.lpoint.position, scene, bvh)) {
      curr_res.weight = 0.0f;
    }
    prev_res.num_candidates =
        min(prev_res.num_candidates, 20 * params.restir_candidates);
    std::vector<restir_reservoir*> reservoirs = {&curr_res, &prev_res};
    reservoir = combine_reservoirs(
        params.restir_vis, params.restir_unbias, point, reservoirs, rng,
        scene, bvh);
  }
  else {
    assert(0 && "Invalid restir_type");
  }

  // check weight
  assert(isfinite(reservoir.weight) && "'reservoir.weight' must be finite");
  if (reservoir.weight == 0.0f || !isfinite(reservoir.weight)) {
    return radiance; 
  }

  // check visibility
  if (!is_point_visible(point.position, reservoir.lpoint.position, scene, bvh)) {
    return radiance;
  }

  // shade
  auto incoming = normalize(reservoir.lpoint.position - point.position);
  radiance += shade_point(point, reservoir, outgoing, incoming);

  // done
  return radiance;
}

static void trace_restir_spatial(
    trace_state* state, const trace_scene* scene, const trace_camera* camera,
    const trace_bvh* bvh, const trace_lights* lights,
    const trace_params& params) {

  // # initial candidates
  parallel_for(state->render.width(), state->render.height(),
      [&] (int i, int j) {
        // initialize
        vec2i ij = vec2i{i, j};

        // trace_sample
        auto ray = sample_camera(camera, ij, state->render.imsize(),
            rand2f(state->rngs[ij]), rand2f(state->rngs[ij]),
            params.tentfilter);

        // intersect next point
        auto intersection = intersect_bvh(bvh, ray);
        if (!intersection.hit) {
          if (!params.envhidden) {
            auto env = eval_environment(scene, ray.d);
            state->accumulation[ij] += fix_radiance(env, params);
          }
          return;
        }

        // prepare shading point
        auto point    = make_shading_point(intersection, -ray.d, scene);

        // accumulate emission
        auto emission = eval_emission(point.emission, point.normal,
                                      point.outgoing);
        state->accumulation[ij] += fix_radiance(emission, params);
        state->render[ij] = fix_radiance(emission, params);

        // handle delta
        if (is_delta(point.bsdf)) { return; }

        // initial candidate
        restir_reservoir& r = state->reservoirs[ij];
        r = make_reservoir(
            params.restir_vis, point, scene, lights, state->rngs[ij],
            params.restir_candidates, bvh);
        // @VIS
        // if (!is_point_visible(point.position, r.lpoint.position, scene, bvh)) {
        //   r.weight = 0.0f;
        // }
      });

  // # temporal reuse
  // parallel_for(state->render.width(), state->render.height(),
  //     [&](int i, int j) {
  //       auto ij = vec2i{i, j};
  //       restir_reservoir curr_res = state->reservoirs[ij];
  //       restir_reservoir prev_res = state->prev_reservoirs[ij];
  //       prev_res.num_candidates =
  //           min(prev_res.num_candidates, 20 * params.restir_candidates);
  //       std::vector<restir_reservoir*> reservoirs = {&curr_res, &prev_res};
  //       state->reservoirs[ij] = combine_reservoirs(
  //           params.restir_vis, params.restir_unbias, curr_res.point,
  //           reservoirs, state->rngs[ij], scene, bvh);
  //     });

  // # spatial reuse
  parallel_for(state->render.width(), state->render.height(),
      [&](int i, int j) {
        auto ij = vec2i{i, j};
        std::vector<restir_reservoir*> reservoirs;
        restir_reservoir* r = &state->reservoirs[ij];
        reservoirs.push_back(r);
        pick_spatial_neighbours(state, ij, reservoirs);
        state->prev_reservoirs[ij] = combine_reservoirs(
            params.restir_vis, params.restir_unbias, r->point,
            reservoirs, state->rngs[ij], scene, bvh);
      });
  std::swap(state->prev_reservoirs, state->reservoirs);

  // # shade
  parallel_for(state->render.width(), state->render.height(),
      [&](int i, int j) {
        // initialize
        auto ij = vec2i{i, j};
        auto& reservoir = state->reservoirs[ij];
        assert(isfinite(reservoir.weight));

        // check reservoir and visibility
        if (reservoir.weight > 0.0f &&
            is_point_visible(reservoir.point.position,
                             reservoir.lpoint.position, scene, bvh)) {
          // accumulate radiance
          auto incoming = normalize(
              reservoir.lpoint.position - reservoir.point.position);
          auto radiance = shade_point(
              reservoir.point, reservoir, reservoir.point.outgoing, incoming);
          state->accumulation[ij] += fix_radiance(radiance, params);
          // @NOACC: uncomment this line
          // state->render[ij] = fix_radiance(radiance, params);
        }
        // @NOACC: comment these lines
        state->samples[ij] += 1;
        state->render[ij] = state->accumulation[ij] / state->samples[ij];
      });
  std::swap(state->prev_reservoirs, state->reservoirs);
}
