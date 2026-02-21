# Decoder to visualize the predictions
with torch.no_grad():
    visual_pred, diff_pred = model.decoder(
        z_pred[..., :-action_embed_dim],
        patch_h,
        patch_w,
        frames_per_latent=tubelet_size
    )

visual_pred = visual_pred.view(B, -1, C, H, W)

for i, (t, f) in enumerate(zip(times, visual_pred[0])):
    axes[1, i].imshow(f.moveaxis(0, -1).detach().clamp(0, 1).cpu().numpy())
    axes[1, i].set_title(f"t = {t+6}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()