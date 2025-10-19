# Exploring the capabilities of the V-JEPA2 model 
For vision task in videos, are these video encoders more capable than image encoders like DINOv2, DINOv3, ... 

What does the encoder's latent space look like?
- [x] PCA without masking.
- [x] PCA with masking, what will we recover?

Can we use this model for video tasks, like point tracking, action classification?
- Can we train a point-tracker by finetuning a tracker head, and maybe some LoRA like weights based on (Kim et al., 2025).

Test action classification example?
- Can we rerun their Something-Something v2 results of their attentive probe?

### References
Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Mojtaba, Komeili, Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Hogan, F. R., Dugas, D., Bojanowski, P., Khalidov, V., … Ballas, N. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (No. arXiv:2506.09985). arXiv. https://doi.org/10.48550/arXiv.2506.09985

Kim, I. H., Cho, S., Huang, J., Yi, J., Lee, J.-Y., & Kim, S. (2025). Exploring Temporally-Aware Features for Point Tracking (No. arXiv:2501.12218). arXiv. https://doi.org/10.48550/arXiv.2501.12218

