# Magnetar GLM component

Initial GLM model component work for [Magnetar](https://github.com/astorise/Magnetar).

## Implemented

This Rust library reads the routing portion of a GLM-4.7-Flash configuration
and computes a single token's grouped, bias-corrected expert selection and
unbiased combination weights from supplied logits. It is a portable reference,
not a Provider kernel. Tests cover the published checkpoint's routing fields,
group selection, bias, normalization including top-1, scaling and invalid input,
and compare synthetic inputs against the upstream PyTorch router.
The integration workflow runs these tests independently and alongside the
shared Magnetar module checks. Tie ordering is deterministic by index and may
differ from PyTorch, whose top-k tie order is not guaranteed.

## Not implemented

There is no GLM model graph, checkpoint loader, weights, Component Model export
or inference. Magnetar's grouped routing and normalization requirements are
tracked in [issue #219](https://github.com/astorise/Magnetar/issues/219);
multi-head latent attention work is tracked in [issue #198](https://github.com/astorise/Magnetar/issues/198).
No GLM checkpoint can be run with this repository yet.
