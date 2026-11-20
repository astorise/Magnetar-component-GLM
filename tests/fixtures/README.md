# GLM-4.7-Flash router configuration subset

`glm47-router-subset.json` contains the routing fields of the public
[`zai-org/GLM-4.7-Flash` configuration](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json),
inspected on 20 November 2026. It is a manually extracted subset, not the
complete original configuration or a weight file. The model card declares
the upstream checkpoint license as MIT.

The analytical tests use independently selected synthetic probabilities to
check grouped selection, unbiased combination scores, normalization and scale.
`glm47-router-pytorch.json` additionally records five synthetic cases produced
by executing the actual `Glm4MoeLiteTopkRouter` class from
[Transformers v5.23.0](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py).
The source file is verified by SHA-256 before its class is loaded; an identity
projection feeds the synthetic logits and no model weights are redistributed.
Regenerate with PyTorch 2.15.0:

```sh
python tests/fixtures/generate_router_reference.py /path/to/modeling_glm4_moe_lite.py
```

These are router tests, not complete model inference.
