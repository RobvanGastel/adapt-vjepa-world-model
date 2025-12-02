# Exploring the capabilities of the V-JEPA2 model 
How does the latent space of V-JEPA2 look like, compared to that of image encoders such as DINOv2, DINOv3, which I explored in earlier repositories.

What does the encoder's latent space look like?
- [x] PCA without masking.
- [x] PCA with masking, what will we recover?

Check out the `Exploration.ipynb` notebook for a more detailed walkthrough of the code and ideas behind it.

_PCA visualizations_ </br>
The V-JEPA2 model takes in two frames merges them to output in the output space as the tubelet size is 2. Give a number of frames of a kitesurfing video below.
![](/assets/frames_kitesurfing.png?raw=true)

When passing for example frame 5, and 6 through the encoder we get out the following latent features when processing them with PCA for visualization purposes. The encoder clearly seperates the kites in both frames.
![](/assets/pca_kitesurfing.png?raw=true)

In this video of a monkey jumping the difference between static and moving objects are more clear. 
![](/assets/frames_monkey.png?raw=true)

In the frames below the tree and the fence are clearly not moving, whereas the encoder clearly encodes the movement of the monkey jumping. This model should therefore be a good starting point to finetune a point tracking model.   
![](/assets/pca_monkey.png?raw=true)

_Downstream Task_ </br>
Can we use this model for video tasks, like point tracking, action classification? It seems that these weights must be well conditioned for downstream video tasks. My initial idea is to add a simple tracker head for videos like done in the paper (Kim et al., 2025). This is still a work in progress.

### References
Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Mojtaba, Komeili, Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Hogan, F. R., Dugas, D., Bojanowski, P., Khalidov, V., … Ballas, N. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (No. arXiv:2506.09985). arXiv. https://doi.org/10.48550/arXiv.2506.09985

Kim, I. H., Cho, S., Huang, J., Yi, J., Lee, J.-Y., & Kim, S. (2025). Exploring Temporally-Aware Features for Point Tracking (No. arXiv:2501.12218). arXiv. https://doi.org/10.48550/arXiv.2501.12218

