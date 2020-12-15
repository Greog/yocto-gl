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
  return point;
}

// Sample lights with respect to area
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

static bool is_point_visible(const vec3f& position, const vec3f& light,
    const trace_scene* scene, const trace_bvh* bvh, float threshold = 0.001) {
  auto incoming            = normalize(light - position);
  auto shadow_ray          = ray3f{position, incoming};
  auto shadow_intersection = intersect_bvh(bvh, shadow_ray);
  if (!shadow_intersection.hit) {
    printf("[warning] Shadow ray hitting nothing!\n");
    return false;
  }

  auto instance = scene->instances[shadow_intersection.instance];
  auto element  = shadow_intersection.element;
  auto uv       = shadow_intersection.uv;
  if (length(light - eval_position(instance, element, uv)) < threshold) {
    return true;
  }
  return false;
}

vec3f restir_eval_incoming(
    const vec3f& position, const restir_light_sample& lsample) {
  if (!lsample.is_environment) {
    return normalize(lsample.position - position);
  } else {
    return lsample.incoming;
  }
}

//
float geometric_term(
    const vec3f& position, const light_point& point, const vec3f& incoming) {
  return abs(dot(point.normal, -incoming)) /
         distance_squared(position, point.position);
}

restir_reservoir make_reservoir(const shading_point& point,
    const vec3f& outgoing, const trace_scene* scene, const trace_lights* lights,
    rng_state& rng, int num_candidates) {
  restir_reservoir res = {};

  auto  sampled_p_hat = 0.0f;
  float w_sum         = 0.0f;

  for (int i = 0; i < num_candidates; i++) {
    // generate candidate
    auto [sample, p] = sample_area_lights(
        scene, lights, rand1f(rng), rand1f(rng), rand2f(rng));
    vec3f incoming = normalize(sample.position - point.position);

    float p_hat = max(
        sample.emission *
        eval_bsdfcos(point.bsdf, point.normal, outgoing, incoming) *
        geometric_term(point.position, sample, incoming));

    float w = p_hat / p;

    // update reservoir
    w_sum += w;
    res.num_candidates += 1;
    if (rand1f(rng) < (w / w_sum)) {
      res.point     = sample;
      sampled_p_hat = p_hat;
    }
  }

  if (sampled_p_hat != 0) {
    res.weight = (1.0f / sampled_p_hat) * (w_sum / res.num_candidates);
  }
  return res;
}

restir_reservoir combine_reservoirs(const vector<restir_reservoir>& reservoirs,
    const shading_point& point, const vec3f& outgoing, rng_state& rng) {
  restir_reservoir result = {};

  auto sampled_p_hat = 0.0f;
  auto w_sum         = 0.0f;
  for (auto& res : reservoirs) {
    vec3f incoming = normalize(res.point.position - point.position);

    float p_hat = max(
        res.point.emission *
        eval_bsdfcos(point.bsdf, point.normal, outgoing, incoming) *
        geometric_term(point.position, res.point, incoming));

    auto w = p_hat * res.weight * res.num_candidates;
    w_sum += w;
    if (rand1f(rng) < (w / w_sum)) {
      result.point  = res.point;
      sampled_p_hat = p_hat;
    }
    result.num_candidates += res.num_candidates;
  }
  if (sampled_p_hat != 0) {
    result.weight = (1.0f / sampled_p_hat) * (w_sum / result.num_candidates);
  }
  return result;
}

static vec3f shade_point(const shading_point& point,
    const restir_reservoir& reservoir, const vec3f& outgoing,
    const vec3f& incoming) {
  return reservoir.point.emission *
         eval_bsdfcos(point.bsdf, point.normal, outgoing, incoming) *
         geometric_term(point.position, reservoir.point, incoming) *
         reservoir.weight;
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
  if (is_delta(point.bsdf)) return radiance;

  auto res = make_reservoir(
      point, outgoing, scene, lights, rng, params.restir_candidates);

  // check visibility
  if (res.weight != 0) {
    if (!is_point_visible(point.position, res.point.position, scene, bvh)) {
      res.weight = 0;
    }
  }

  // auto& reservoir = res; // reuse inactive
  auto& reservoir = state->reservoirs[ij];
  reservoir       = combine_reservoirs({reservoir, res}, point, outgoing, rng);

  if (reservoir.weight != 0) {
    auto incoming = normalize(reservoir.point.position - point.position);
    radiance += shade_point(point, reservoir, outgoing, incoming);
  }

  return radiance;
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

// combine the first reservoir with all the others
// // the output reservoir shall not be in the array
// // probably can be faster
// static vec3f restir_combine_reservoirs(old_restir_reservoir* output,
//     old_restir_reservoir** reservoirs, int count, rng_state& rng) {
//   float                 weight_sum       = 0.0f;
//   old_restir_reservoir* chosen_r         = reservoirs[0];
//   old_restir_reservoir* curr_r           = reservoirs[0];
//   uint64_t              candidates_count = 0;

//   for (int i = 0; i < count; i++) {
//     old_restir_reservoir* r = reservoirs[i];
//     vec3f incoming = restir_eval_incoming(curr_r->position, r->lsample);
//     float weight   = max(eval_bsdfcos(curr_r->bsdf, curr_r->normal,
//                            curr_r->outgoing, incoming) *
//                        r->lsample.emission) *
//                    r->weight * r->candidates_count;
//     weight_sum += weight;
//     candidates_count += r->candidates_count;
//     if (rand1f(rng) < (weight / weight_sum)) {
//       chosen_r = r;
//     }
//   }

//   output->candidates_count = candidates_count;
//   output->lsample          = chosen_r->lsample;
//   output->position         = curr_r->position;
//   output->normal           = curr_r->normal;
//   output->outgoing         = curr_r->outgoing;
//   output->bsdf             = curr_r->bsdf;
//   output->is_valid         = true;

//   uint64_t Z = 0;
//   for (int i = 0; i < count; i++) {
//     old_restir_reservoir* r = reservoirs[i];
//     vec3f incoming = restir_eval_incoming(r->position, output->lsample);
//     float pdf = max(eval_bsdfcos(r->bsdf, r->normal, r->outgoing, incoming) *
//                     output->lsample.emission);
//     if (pdf > 0) {
//       Z += r->candidates_count;
//     }
//   }
//   float m = 1.0f / Z;

//   vec3f incoming = restir_eval_incoming(output->position, output->lsample);
//   vec3f bsdfcos  = eval_bsdfcos(
//       output->bsdf, output->normal, output->outgoing, incoming);
//   float pdf      = max(bsdfcos * output->lsample.emission);
//   output->weight = (1.0f / pdf) * (m * weight_sum);
//   return bsdfcos;
// }
