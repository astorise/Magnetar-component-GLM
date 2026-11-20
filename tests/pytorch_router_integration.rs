use magnetar_component_glm::RouterConfig;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    config: serde_json::Value,
    logits: Vec<f64>,
    bias: Vec<f64>,
    expected: Vec<Expected>,
}

#[derive(Deserialize)]
struct Expected {
    expert: usize,
    weight: f64,
}

#[test]
fn synthetic_inputs_match_upstream_pytorch_glm_router() {
    let fixture: Fixture =
        serde_json::from_slice(include_bytes!("fixtures/glm47-router-pytorch.json")).unwrap();
    assert_eq!(fixture.cases.len(), 5);
    for case in fixture.cases {
        let config =
            RouterConfig::from_huggingface_config(&serde_json::to_vec(&case.config).unwrap())
                .unwrap();
        let mut actual = config.route(&case.logits, &case.bias).unwrap();
        actual.sort_by_key(|item| item.expert);
        assert_eq!(actual.len(), case.expected.len(), "{}", case.name);
        for (actual, expected) in actual.iter().zip(&case.expected) {
            assert_eq!(actual.expert, expected.expert, "{}", case.name);
            assert!(
                (actual.weight - expected.weight).abs() < 1e-6,
                "{}: expert {} weight {} != {}",
                case.name,
                actual.expert,
                actual.weight,
                expected.weight
            );
        }
    }
}
