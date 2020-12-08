
struct light_point {
  vec3f position = {};
  vec3f normal   = {};
  vec3f emission = {};
};
struct shading_point {
  vec3f      position = {};
  vec3f      normal   = {};
  vec3f      emission = {};
  trace_bsdf bsdf     = {};
  // float opacity = 1;
};

inline shading_point make_shading_point(const bvh_intersection& intersection,
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

// Sample lights wrt area
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

static restir_light_sample sample_lights_restir(const trace_scene* scene,
    const trace_lights* lights, const vec3f& position, float rl, float rel,
    const vec2f& ruv, const vec3f& outgoing) {
  restir_light_sample lsample  = {};
  auto                light_id = sample_uniform((int)lights->lights.size(), rl);
  auto                light    = lights->lights[light_id];
  vec3f               incoming;
  vec3f               emission;
  if (light->instance != nullptr) {
    lsample.is_environment = false;
    auto instance          = light->instance;
    auto element           = sample_discrete_cdf(light->elements_cdf, rel);
    auto uv = (!instance->shape->triangles.empty()) ? sample_triangle(ruv)
                                                    : ruv;
    auto normal    = eval_shading_normal(instance, element, uv, outgoing);
    auto lposition = eval_position(light->instance, element, uv);
    emission       = eval_emission(
        eval_emission(instance, element, uv, normal, outgoing), normal,
        outgoing);
    lsample.position = lposition;
  } else if (light->environment != nullptr) {
    lsample.is_environment = true;
    auto environment       = light->environment;
    if (environment->emission_tex != nullptr) {
      auto emission_tex = environment->emission_tex;
      auto idx          = sample_discrete_cdf(light->elements_cdf, rel);
      auto size         = texture_size(emission_tex);
      auto uv           = vec2f{
          ((idx % size.x) + 0.5f) / size.x, ((idx / size.x) + 0.5f) / size.y};
      incoming         = transform_direction(environment->frame,
          {cos(uv.x * 2 * pif) * sin(uv.y * pif), cos(uv.y * pif),
              sin(uv.x * 2 * pif) * sin(uv.y * pif)});
      lsample.incoming = incoming;
    } else {
      incoming = sample_sphere(ruv);
    }
    emission = eval_environment(scene, incoming);
  } else {
    incoming = zero3f;
  }
  lsample.emission = emission;
  return lsample;
}

vec3f restir_eval_incoming(
    const vec3f& position, const restir_light_sample& lsample) {
  if (!lsample.is_environment) {
    return normalize(lsample.position - position);
  } else {
    return lsample.incoming;
  }
}

// combine the first reservoir with all the others
// the output reservoir shall not be in the array
// probably can be faster
static vec3f restir_combine_reservoirs(restir_reservoir* output,
    restir_reservoir** reservoirs, int count, rng_state& rng) {
  float             weight_sum       = 0.0f;
  restir_reservoir* chosen_r         = reservoirs[0];
  restir_reservoir* curr_r           = reservoirs[0];
  uint64_t          candidates_count = 0;

  for (int i = 0; i < count; i++) {
    restir_reservoir* r = reservoirs[i];
    vec3f incoming      = restir_eval_incoming(curr_r->position, r->lsample);
    float weight        = max(eval_bsdfcos(curr_r->bsdf, curr_r->normal,
                           curr_r->outgoing, incoming) *
                       r->lsample.emission) *
                   r->weight * r->candidates_count;
    weight_sum += weight;
    candidates_count += r->candidates_count;
    if (rand1f(rng) < (weight / weight_sum)) {
      chosen_r = r;
    }
  }

  output->candidates_count = candidates_count;
  output->lsample          = chosen_r->lsample;
  output->position         = curr_r->position;
  output->normal           = curr_r->normal;
  output->outgoing         = curr_r->outgoing;
  output->bsdf             = curr_r->bsdf;
  output->is_valid         = true;

  uint64_t Z = 0;
  for (int i = 0; i < count; i++) {
    restir_reservoir* r = reservoirs[i];
    vec3f incoming      = restir_eval_incoming(r->position, output->lsample);
    float pdf = max(eval_bsdfcos(r->bsdf, r->normal, r->outgoing, incoming) *
                    output->lsample.emission);
    if (pdf > 0) {
      Z += r->candidates_count;
    }
  }
  float m = 1.0f / Z;

  vec3f incoming = restir_eval_incoming(output->position, output->lsample);
  vec3f bsdfcos  = eval_bsdfcos(
      output->bsdf, output->normal, output->outgoing, incoming);
  float pdf      = max(bsdfcos * output->lsample.emission);
  output->weight = (1.0f / pdf) * (m * weight_sum);
  return bsdfcos;
}

float geometric_term(
    const vec3f& position, const light_point& point, const vec3f& incoming) {
  return abs(dot(point.normal, -incoming)) /
         distance_squared(position, point.position);
}

static vec3f f_trace_restir(const trace_scene* scene, const trace_bvh* bvh,
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
    } else {
      return zero3f;
    }
  }
  hit = true;

  // prepare shading point
  auto outgoing = -ray.d;
  auto instance = scene->instances[intersection.instance];
  auto element  = intersection.element;
  auto uv       = intersection.uv;
  auto position = eval_position(instance, element, uv);
  auto normal   = eval_shading_normal(instance, element, uv, outgoing);
  auto emission = eval_emission(instance, element, uv, normal, outgoing);
  // auto opacity  = eval_opacity(instance, element, uv, normal, outgoing);
  auto bsdf = eval_bsdf(instance, element, uv, normal, outgoing);

  // accumulate emission
  radiance += eval_emission(emission, normal, outgoing);

  // handle delta
  if (is_delta(bsdf)) return radiance;

  // sample direct light
  const int candidates_count = 16;
  float     weight_sum       = 0.0f;
  // reservoir of size 1 (= 1 sample)
  vec3f sampled_incoming = zero3f;
  vec3f sampled_emission = zero3f;
  float gterm            = 1;

  for (int i = 1; i <= candidates_count; i++) {
    auto [candidate_point, pdf] = sample_area_lights(
        scene, lights, rand1f(rng), rand1f(rng), rand2f(rng));
    auto candidate_incoming = normalize(candidate_point.position - position);
    // float pdf                                     = sample_lights_pdf(
    // scene, bvh, lights, position, candidate_incoming);
    vec3f bsdfcos = eval_bsdfcos(bsdf, normal, outgoing, candidate_incoming);
    float candidate_weight = (max(bsdfcos * candidate_point.emission)) / pdf;
    weight_sum += candidate_weight;
    if (rand1f(rng) < (candidate_weight / weight_sum)) {
      sampled_incoming = candidate_incoming;
      sampled_emission = candidate_point.emission;
      gterm = geometric_term(position, candidate_point, candidate_incoming);
    }
  }

  vec3f incoming = sampled_incoming;
  vec3f bsdfcos  = eval_bsdfcos(bsdf, normal, outgoing, incoming);
  vec3f weight   = bsdfcos * ((1 / max(bsdfcos * sampled_emission)) *
                               ((1.0f / candidates_count) * weight_sum));
  weight *= gterm;
  // check weight
  if (weight == zero3f || !isfinite(weight)) return radiance;

  // shadow ray
  auto shadow_ray          = ray3f{position, incoming};
  auto shadow_intersection = intersect_bvh(bvh, shadow_ray);
  if (!shadow_intersection.hit) {
    if (params.envhidden) {
      return radiance;
    } else {
      radiance += weight * eval_environment(scene, shadow_ray.d);
    }
  } else {
    auto outgoing = -shadow_ray.d;
    auto instance = scene->instances[shadow_intersection.instance];
    auto element  = shadow_intersection.element;
    auto uv       = shadow_intersection.uv;
    // auto position = eval_position(instance, element, uv);
    auto normal   = eval_shading_normal(instance, element, uv, outgoing);
    auto emission = eval_emission(instance, element, uv, normal, outgoing);
    // auto opacity  = eval_opacity(instance, element, uv, normal, outgoing);
    // auto bsdf     = eval_bsdf(instance, element, uv, normal, outgoing);
    radiance += weight * eval_emission(emission, normal, outgoing);
  }
  return radiance;
}

pair<light_point, vec3f> sample_lights_reservoir(const shading_point& point,
    const vec3f& outgoing, const trace_scene* scene, const trace_lights* lights,
    rng_state& rng, int num_candidates) {
  light_point light_point = {};
  vec3f       result      = zero3f;
  float       w_sum       = 0.0f;

  for (int i = 0; i < num_candidates; i++) {
    auto [sample, p] = sample_area_lights(
        scene, lights, rand1f(rng), rand1f(rng), rand2f(rng));

    vec3f incoming  = normalize(sample.position - point.position);
    vec3f integrand = sample.emission;
    integrand *= eval_bsdfcos(point.bsdf, point.normal, outgoing, incoming);
    integrand *= geometric_term(point.position, sample, incoming);

    float w = max(integrand) / p;
    w_sum += w;

    if (rand1f(rng) < (w / w_sum)) {
      light_point = sample;
      result      = integrand;
    }
  }

  if (result != zero3f) {
    result /= max(result);  // divide by p_hat
    result *= w_sum / num_candidates;
  }

  return {light_point, result};
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
    } else {
      return zero3f;
    }
  }
  hit = true;

  // prepare shading point
  auto outgoing = -ray.d;
  auto point    = make_shading_point(intersection, outgoing, scene);

  // accumulate emission
  radiance += point.emission;

  // handle delta
  if (is_delta(point.bsdf)) return point.emission;

  auto [light_point, integrand] = sample_lights_reservoir(
      point, outgoing, scene, lights, rng, params.restir_candidates);

  // check visibility
  if (integrand != zero3f) {
    if (!is_point_visible(point.position, light_point.position, scene, bvh)) {
      return point.emission;
    }
  }

  return point.emission + integrand;
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
  auto [light_point, pdf] = sample_area_lights(
      scene, lights, rand1f(rng), rand1f(rng), rand2f(rng));
  auto incoming = normalize(light_point.position - point.position);
  auto bsdfcos  = eval_bsdfcos(point.bsdf, point.normal, outgoing, incoming);
  auto weight   = bsdfcos / pdf;

  // check weight
  if (weight == zero3f || !isfinite(weight))
    return {radiance.x, radiance.y, radiance.z, 1};

  // check visibility
  if (!is_point_visible(point.position, light_point.position, scene, bvh)) {
    return {radiance.x, radiance.y, radiance.z, 1};
  }

  // TODO(giacomo): refactor this into a function.
  auto geometric_term = abs(dot(light_point.normal, -incoming)) /
                        distance_squared(point.position, light_point.position);
  weight *= geometric_term;
  radiance += weight * light_point.emission;

  return {radiance.x, radiance.y, radiance.z, 1};
}
