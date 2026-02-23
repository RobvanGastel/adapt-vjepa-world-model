# Exploring the capabilities of the V-JEPA2 model 
How does the latent space of V-JEPA2 look like, compared to that of image encoders such as DINOv2, DINOv3, which I explored in earlier repositories. Also potentially these SSL pre-trained models are ways to train World Models without massive amounts of compute.

Investigate the latent space of the V-JEPA2 model by:
- [x] PCA without masking.
- [x] PCA with masking, what will we recover?

Check out the `Exploration.ipynb` notebook for a more detailed walkthrough of the code and ideas behind it.

- [x] Compare transition model on latent space predictions of VJEPA2 with DINO. Currently, skipping other encoders.
    - [x] Add a decoder for visualization purposes.
- [x] Generate a better dataset, option for simple RGB frame environment.
    - [x] Balancing a pendulum, and secondly include the actions.
- [x] Add option for actions with MPC and CEM.

Check out the `World Model.ipynb` notebook to test the planning of the model. However, the continuous action space is a lot more difficult than the original paper with discrete action space.

(WIP) Fixed autoregressive latent predictor! Again testing the CEM MPC loop.

## Setup
Install the packages using the `requirements.txt` file.

```bash
# using conda
conda create --name jepa python=3.11
conda activate jepa

# Run the training code, adjust the argparse flags
python train_world_model.py
# Test the planning of the model on the pendulum environment
python test_planning.py
```

## Model Weights

The model is split into components, the [action embedding network](https://drive.google.com/file/d/1VzL1d_D9b4Laix5NA0NgIYh2D6YzsJDI/view), [ViT latent predictor](https://drive.google.com/file/d/1JdZ5Qg7V3f8oZqtlYF1JJPsKLNz-jRgl/view) for future state prediction, and the [decoder](https://drive.google.com/file/d/1VYO7AEgwW8DteAj_GQgQ-oEAjK7BpRFj/view) for visualizing the latents.

Finally, put these into the output folder so the networks can easily be loaded back in.

### PCA visualizations </br>
The V-JEPA2 model takes in two frames merges them to output in the output space as the tubelet size is 2. Give a number of frames of a kitesurfing video below.
![](/assets/frames_kitesurfing.png?raw=true)

When passing for example frame 5, and 6 through the encoder we get out the following latent features when processing them with PCA for visualization purposes. The encoder clearly seperates the kites in both frames.
![](/assets/pca_kitesurfing.png?raw=true)

<!-- In this video of a monkey jumping the difference between static and moving objects are more clear. 
![](/assets/frames_monkey.png?raw=true)

In the frames below the tree and the fence are clearly not moving, whereas the encoder clearly encodes the movement of the monkey jumping. This model should therefore be a good starting point to finetune next frame prediction.   
![](/assets/pca_monkey.png?raw=true) -->

### World Model
These are the outputs of training the future latent state predictor and decoder on top of the Pendulum environment. As you can see it predicts the first 3 states pretty accurately.
![](/assets/future_state_predictions.png?raw=true)

As for the latent state comparison between the predictor and the encoder. These are also comparable.
![](/assets/predictor_latent_space.png?raw=true)


### References
Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Mojtaba, Komeili, Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Hogan, F. R., Dugas, D., Bojanowski, P., Khalidov, V., … Ballas, N. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (No. arXiv:2506.09985). arXiv. https://doi.org/10.48550/arXiv.2506.09985

Kim, I. H., Cho, S., Huang, J., Yi, J., Lee, J.-Y., & Kim, S. (2025). Exploring Temporally-Aware Features for Point Tracking (No. arXiv:2501.12218). arXiv. https://doi.org/10.48550/arXiv.2501.12218

Zhou, G., Pan, H., LeCun, Y., & Pinto, L. (2025). DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (No. arXiv:2411.04983). arXiv. https://doi.org/10.48550/arXiv.2411.04983
