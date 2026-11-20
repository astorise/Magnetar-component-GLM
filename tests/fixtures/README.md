# GLM-4.7-Flash router configuration subset

`glm47-router-subset.json` contains the routing fields of the public
[`zai-org/GLM-4.7-Flash` configuration](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json),
inspected on 20 November 2026. It is a manually extracted subset, not the
complete original configuration or a weight file. The model card declares
the upstream checkpoint license as MIT.

The tests use independently selected synthetic probabilities and analytical
sigmoid/logit values to check grouped selection, unbiased combination scores,
normalization and scale. They do not compare against PyTorch or run the model.
