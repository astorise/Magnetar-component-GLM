use magnetar_component_glm::RouterConfig;

fn custom(groups: usize, selected_groups: usize, top_k: usize, normalize: bool) -> RouterConfig {
    let json = serde_json::json!({
        "model_type": "glm4_moe_lite", "topk_method": "noaux_tc",
        "n_routed_experts": 8, "n_group": groups, "topk_group": selected_groups,
        "num_experts_per_tok": top_k, "norm_topk_prob": normalize,
        "routed_scaling_factor": 1.8,
    });
    RouterConfig::from_huggingface_config(&serde_json::to_vec(&json).unwrap()).unwrap()
}

fn logit(probability: f64) -> f64 {
    (probability / (1.0 - probability)).ln()
}

#[test]
fn published_glm47_dimensions_route_bias_selected_experts_with_unbiased_weights() {
    let router =
        RouterConfig::from_huggingface_config(include_bytes!("fixtures/glm47-router-subset.json"))
            .unwrap();
    assert_eq!(router.expert_count(), 64);
    let mut logits = vec![logit(0.1); 64];
    logits[0] = logit(0.2);
    logits[1] = logit(0.3);
    logits[2] = logit(0.4);
    logits[3] = logit(0.5);
    logits[4] = logit(0.6);
    let mut bias = vec![0.0; 64];
    bias[0] = 1.0;
    let result = router.route(&logits, &bias).unwrap();
    assert_eq!(
        result.iter().map(|x| x.expert).collect::<Vec<_>>(),
        vec![0, 4, 3, 2]
    );
    let selected_sum = 0.2 + 0.6 + 0.5 + 0.4;
    for (item, expected) in result.iter().zip([0.2, 0.6, 0.5, 0.4]) {
        assert!((item.weight - expected / selected_sum * 1.8).abs() < 1e-12);
    }
}

#[test]
fn group_selection_uses_sum_of_two_biased_scores_not_single_best() {
    let router = custom(2, 1, 2, true);
    // Group 0 has the highest individual score, but group 1 has the higher
    // sum of its best two scores. Bias then changes which experts it selects.
    let probabilities = [0.99, 0.10, 0.10, 0.10, 0.70, 0.69, 0.30, 0.20];
    let logits: Vec<_> = probabilities.into_iter().map(logit).collect();
    let bias = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0];
    let result = router.route(&logits, &bias).unwrap();
    assert_eq!(
        result.iter().map(|x| x.expert).collect::<Vec<_>>(),
        vec![6, 4]
    );
    assert!((result[0].weight - 0.3 / (0.3 + 0.7) * 1.8).abs() < 1e-12);
    assert!((result[1].weight - 0.7 / (0.3 + 0.7) * 1.8).abs() < 1e-12);
}

#[test]
fn normalization_applies_to_top_one_and_extreme_negative_logits_are_safe() {
    let router = custom(2, 1, 1, true);
    let result = router
        .route(&[1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0], &[0.0; 8])
        .unwrap();
    assert_eq!(result[0].expert, 0);
    assert!((result[0].weight - 1.8).abs() < 1e-12);
    let zeros = router.route(&[-1000.0; 8], &[0.0; 8]).unwrap();
    assert_eq!(zeros[0].expert, 0); // deterministic local tie policy
    assert_eq!(zeros[0].weight, 0.0);
}

#[test]
fn disabled_normalization_retains_unbiased_sigmoid_and_scale() {
    let router = custom(2, 1, 1, false);
    let result = router
        .route(&[logit(0.2); 8], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .unwrap();
    assert_eq!(result[0].expert, 0);
    assert!((result[0].weight - 0.36).abs() < 1e-12);
}

#[test]
fn invalid_inputs_fail_instead_of_changing_routing_semantics() {
    let router = custom(2, 1, 2, true);
    assert!(router.route(&[0.0; 7], &[0.0; 8]).is_err());
    assert!(router.route(&[f64::NAN; 8], &[0.0; 8]).is_err());
    assert!(router.route(&[0.0; 8], &[f64::INFINITY; 8]).is_err());
    for config in [
        br#"{"model_type":"glm4_moe_lite","topk_method":"noaux_tc","n_routed_experts":8,"n_group":3,"topk_group":1,"num_experts_per_tok":2,"norm_topk_prob":true,"routed_scaling_factor":1.8}"#.as_slice(),
        br#"{"model_type":"glm4_moe_lite","topk_method":"noaux_tc","n_routed_experts":8,"n_group":2,"topk_group":1,"num_experts_per_tok":5,"norm_topk_prob":true,"routed_scaling_factor":1.8}"#,
        br#"{"model_type":"glm4_moe_lite","topk_method":"other","n_routed_experts":8,"n_group":2,"topk_group":1,"num_experts_per_tok":2,"norm_topk_prob":true,"routed_scaling_factor":1.8}"#,
    ] {
        assert!(RouterConfig::from_huggingface_config(config).is_err());
    }
}
