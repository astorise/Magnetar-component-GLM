//! GLM-4.7-Flash routing reference for Magnetar.
//!
//! This computes a single token's expert selection from supplied router
//! logits and correction biases. It is a portable reference for integration,
//! not a Model Component graph, Provider kernel, or model inference engine.

use serde::Deserialize;
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

const MAX_EXPERTS: usize = 65_536;

/// Grouped top-k routing parameters from a GLM-4-MoE-Lite checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct RouterConfig {
    experts: usize,
    groups: usize,
    selected_groups: usize,
    top_k: usize,
    normalize: bool,
    scale: f64,
}

#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    topk_method: String,
    n_routed_experts: usize,
    n_group: usize,
    topk_group: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
}

impl RouterConfig {
    /// Parse the routing subset of a `glm4_moe_lite` Hugging Face config.
    /// This does not validate other architecture fields or enable inference.
    pub fn from_huggingface_config(bytes: &[u8]) -> Result<Self, RoutingError> {
        let config: RawConfig = serde_json::from_slice(bytes)
            .map_err(|error| RoutingError(format!("invalid GLM router config: {error}")))?;
        if config.model_type != "glm4_moe_lite" || config.topk_method != "noaux_tc" {
            return Err(RoutingError(
                "expected glm4_moe_lite and noaux_tc routing".into(),
            ));
        }
        if !(2..=MAX_EXPERTS).contains(&config.n_routed_experts)
            || config.n_group == 0
            || !config.n_routed_experts.is_multiple_of(config.n_group)
            || config.n_routed_experts / config.n_group < 2
            || config.topk_group == 0
            || config.topk_group > config.n_group
        {
            return Err(RoutingError("invalid expert or group dimensions".into()));
        }
        let available = config.topk_group * (config.n_routed_experts / config.n_group);
        if config.num_experts_per_tok == 0 || config.num_experts_per_tok > available {
            return Err(RoutingError(
                "top-k exceeds the selected groups' capacity".into(),
            ));
        }
        if !config.routed_scaling_factor.is_finite() || config.routed_scaling_factor <= 0.0 {
            return Err(RoutingError(
                "routed scaling factor must be finite and positive".into(),
            ));
        }
        Ok(Self {
            experts: config.n_routed_experts,
            groups: config.n_group,
            selected_groups: config.topk_group,
            top_k: config.num_experts_per_tok,
            normalize: config.norm_topk_prob,
            scale: config.routed_scaling_factor,
        })
    }

    /// The number of routed experts expected in each logits and bias vector.
    pub fn expert_count(&self) -> usize {
        self.experts
    }

    /// Select experts for one token. Ties are resolved by ascending index;
    /// PyTorch top-k does not guarantee the same tie order. Returned entries
    /// pair each selected expert with its own unbiased combination weight.
    pub fn route(&self, logits: &[f64], bias: &[f64]) -> Result<Vec<ExpertWeight>, RoutingError> {
        if logits.len() != self.experts || bias.len() != self.experts {
            return Err(RoutingError(
                "logits and bias must have one value per expert".into(),
            ));
        }
        if logits.iter().chain(bias).any(|value| !value.is_finite()) {
            return Err(RoutingError(
                "router logits and biases must be finite".into(),
            ));
        }
        let scores: Vec<f64> = logits.iter().map(|&value| sigmoid(value)).collect();
        let choices: Vec<f64> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
        let group_size = self.experts / self.groups;
        let mut groups = Vec::with_capacity(self.groups);
        for (group, values) in choices.chunks_exact(group_size).enumerate() {
            let mut highest = f64::NEG_INFINITY;
            let mut second = f64::NEG_INFINITY;
            for &value in values {
                if value > highest {
                    second = highest;
                    highest = value;
                } else if value > second {
                    second = value;
                }
            }
            let score = highest + second;
            if !score.is_finite() {
                return Err(RoutingError("group selection score overflowed".into()));
            }
            groups.push((group, score));
        }
        groups.sort_by(|a, b| descending_score(a.1, b.1).then(a.0.cmp(&b.0)));
        let mut candidates = Vec::with_capacity(self.selected_groups * group_size);
        for &(group, _) in groups.iter().take(self.selected_groups) {
            let first = group * group_size;
            candidates.extend((first..first + group_size).map(|index| (index, choices[index])));
        }
        candidates.sort_by(|a, b| descending_score(a.1, b.1).then(a.0.cmp(&b.0)));
        let mut selected: Vec<_> = candidates
            .into_iter()
            .take(self.top_k)
            .map(|(expert, _)| ExpertWeight {
                expert,
                weight: scores[expert],
            })
            .collect();
        if self.normalize {
            let denominator = selected.iter().map(|item| item.weight).sum::<f64>() + 1e-20;
            for item in &mut selected {
                item.weight /= denominator;
            }
        }
        for item in &mut selected {
            item.weight *= self.scale;
        }
        Ok(selected)
    }
}

fn descending_score(a: f64, b: f64) -> Ordering {
    b.total_cmp(&a)
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

/// An expert index and its unrounded combination weight.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertWeight {
    pub expert: usize,
    pub weight: f64,
}

/// A malformed, unsupported, or numerically invalid routing request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingError(String);

impl fmt::Display for RoutingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for RoutingError {}
