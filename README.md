# Exploring the capabilities of the V-JEPA2 model 
How does the latent space of V-JEPA2 look like, compared to that of image encoders such as DINOv2, DINOv3, which I explored in earlier repositories.

What does the encoder's latent space look like?
- [x] PCA without masking.
- [x] PCA with masking, what will we recover?

Check out the `Exploration.ipynb` notebook for a more detailed walkthrough of the code and ideas behind it.

Are there benefits to the temporal information it learns versus other encoders such as DINOv2, DINOV3.
- [ ] Compare transition model on latent space predictions of VJEPA2 with DINO
    - [x] Add a decoder for visualization purposes.

(WIP) Currently, a working next frame and VQ VAE decoder. However, no comparison yet with DINO encoders.

### PCA visualizations </br>
The V-JEPA2 model takes in two frames merges them to output in the output space as the tubelet size is 2. Give a number of frames of a kitesurfing video below.
![](/assets/frames_kitesurfing.png?raw=true)

When passing for example frame 5, and 6 through the encoder we get out the following latent features when processing them with PCA for visualization purposes. The encoder clearly seperates the kites in both frames.
![](/assets/pca_kitesurfing.png?raw=true)

In this video of a monkey jumping the difference between static and moving objects are more clear. 
![](/assets/frames_monkey.png?raw=true)

In the frames below the tree and the fence are clearly not moving, whereas the encoder clearly encodes the movement of the monkey jumping. This model should therefore be a good starting point to finetune next frame prediction.   
![](/assets/pca_monkey.png?raw=true)

### World Model



### References
Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Mojtaba, Komeili, Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Hogan, F. R., Dugas, D., Bojanowski, P., Khalidov, V., … Ballas, N. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (No. arXiv:2506.09985). arXiv. https://doi.org/10.48550/arXiv.2506.09985

Kim, I. H., Cho, S., Huang, J., Yi, J., Lee, J.-Y., & Kim, S. (2025). Exploring Temporally-Aware Features for Point Tracking (No. arXiv:2501.12218). arXiv. https://doi.org/10.48550/arXiv.2501.12218

Zhou, G., Pan, H., LeCun, Y., & Pinto, L. (2025). DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning (No. arXiv:2411.04983). arXiv. https://doi.org/10.48550/arXiv.2411.04983
