#include <assert.h>

static shading_point make_shading_point(const bvh_intersection& intersection,
    const vec3f& outgoing, const trace_scene* scene) {
  auto instance  = scene->instances[intersection.instance];
  auto element   = intersection.element;
  auto uv        = intersection.uv;
  auto point     = shading_point{};
  point.position = eval_position(instance, element, uv);
  point.normal   = eval_shading_normal(instance, element, uv, outgoing);
  point.emission = eval_emission(instance, element, uv, point.normal, outgoing);
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
    printf("[warning] Shadow ray hitting nothing!\n");
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
    assert(0 && "environments not supported for now\n");
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

static restir_reservoir combine_reservoirs_biased(
    const shading_point& point, const vec3f& outgoing,
    const vector<restir_reservoir*>& reservoirs, rng_state& rng) {
  restir_reservoir        res;
  float                   w_sum           = 0.0f;
  restir_reservoir*       sampled_res     = nullptr;
  float                   sampled_p_hat_q = 0.0f;

  for (auto r : reservoirs) {
    vec3f incoming = normalize(r->lpoint.position - point.position);
    vec3f bsdfcos  = eval_bsdfcos(point.bsdf, point.normal, outgoing, incoming);
    float gterm    = geometric_term(
        point.position, r->lpoint.position, r->lpoint.normal, incoming);
    float p_hat_q  = max(bsdfcos * r->lpoint.emission) * gterm;
    float w        = p_hat_q * r->weight * r->num_candidates;

    w_sum += w;
    res.num_candidates += r->num_candidates;
    if (rand1f(rng) < (w / w_sum)) {
      sampled_res     = r;
      sampled_p_hat_q = p_hat_q;
    }
  }

  if (sampled_res != nullptr) {
    res.lpoint = sampled_res->lpoint;
    res.weight = (1.0f / sampled_p_hat_q) * (1.0f / res.num_candidates) * w_sum;
  }

  return res;
}

static restir_reservoir make_reservoir(const shading_point& point,
    const vec3f& outgoing, const trace_scene* scene, const trace_lights* lights,
    rng_state& rng, int num_candidates) {
  restir_reservoir res           = {};
  float            w_sum         = 0.0f;
  float            sampled_p_hat = 0.0f;

  // generate initial candidates
  for (int i = 0; i < num_candidates; i++) {
    auto [candidate, candidate_pdf] = sample_area_lights(
        scene, lights, rand1f(rng), rand1f(rng), rand2f(rng));
    vec3f incoming = normalize(candidate.position - point.position);
    vec3f bsdfcos  = eval_bsdfcos(point.bsdf, point.normal, outgoing, incoming);
    float gterm    = geometric_term(
      point.position, candidate.position, candidate.normal, incoming);
    float p_hat    = max(bsdfcos * candidate.emission) * gterm;
    float w        = p_hat / candidate_pdf;

    // update reservoir
    w_sum += w;
    res.num_candidates += 1;
    if (rand1f(rng) < (w / w_sum)) {
      res.lpoint = candidate;
      sampled_p_hat = p_hat;
    }
  }

  if (sampled_p_hat != 0.0f) {
    res.weight = (1.0f / sampled_p_hat) * (1.0f / num_candidates) * w_sum;
  }

  return res;
}

static vec3f shade_point(const shading_point& point,
    const restir_reservoir& reservoir, const vec3f& outgoing,
    const vec3f& incoming) {
  vec3f bsdfcos = eval_bsdfcos(point.bsdf, point.normal, outgoing, incoming);
  float gterm = geometric_term(
      point.position, reservoir.lpoint.position, reservoir.lpoint.normal, incoming);
  return reservoir.weight * reservoir.lpoint.emission * bsdfcos * gterm;
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
  auto reservoir = make_reservoir(
      point, outgoing, scene, lights, rng, params.restir_candidates);
  auto incoming  = normalize(reservoir.lpoint.position - point.position);

  // check weight
  if (reservoir.weight == 0.0f || !isfinite(reservoir.weight)) {
    return radiance; 
  }

  // check visibility
  if (!is_point_visible(
        point.position, reservoir.lpoint.position, scene, bvh)) {
    return radiance;
  }

  radiance += shade_point(point, reservoir, outgoing, incoming);

  return radiance;
}
